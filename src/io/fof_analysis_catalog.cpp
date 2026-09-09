#include "cosmo_nbody/io/fof_analysis_catalog.hpp"

#include "cosmo_nbody/core/id_uniqueness.hpp"
#include "cosmo_nbody/io/bounded_hdf5_string.hpp"
#include "cosmo_nbody/io/hdf5_handle.hpp"
#include "cosmo_nbody/io/output_schema.hpp"
#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/rounded_product_relation.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmo_nbody::io {
namespace {

bool finite_vec3(const core::Vec3& value) {
    return std::isfinite(value.x)
        && std::isfinite(value.y)
        && std::isfinite(value.z);
}

bool nearly_equal(
    core::Real lhs,
    core::Real rhs,
    core::Real relative_tolerance =
        core::Real{512.0} * std::numeric_limits<core::Real>::epsilon()) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)
        || !std::isfinite(relative_tolerance)
        || relative_tolerance < 0.0) {
        return false;
    }
    const core::Real scale = std::max(std::abs(lhs), std::abs(rhs));
    if (scale == 0.0) return true;
    return std::abs(lhs - rhs) <= relative_tolerance * scale;
}

std::uint64_t checked_u64(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds uint64");
        }
    }
    return static_cast<std::uint64_t>(value);
}

std::size_t checked_size(std::uint64_t value, const char* label) {
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (value > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds size_t");
        }
    }
    return static_cast<std::size_t>(value);
}

hsize_t checked_hsize(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(hsize_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<hsize_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds hsize_t");
        }
    }
    return static_cast<hsize_t>(value);
}

ProductLineage parse_lineage(const std::string& value) {
    if (value == "direct_simulation") return ProductLineage::DirectSimulation;
    if (value == "derived_analysis") return ProductLineage::DerivedAnalysis;
    throw std::runtime_error("Unsupported FoF analysis product lineage: " + value);
}

std::string checked_lineage(ProductLineage lineage) {
    if (lineage == ProductLineage::Unknown) {
        throw std::invalid_argument(
            "Split FoF analysis catalog requires non-unknown lineage");
    }
    const std::string value = lineage_to_string(lineage);
    if (value.empty() || value == "unknown") {
        throw std::invalid_argument(
            "Split FoF analysis catalog lineage is not serializable");
    }
    return value;
}

void write_attr(
    hid_t location,
    const std::string& name,
    hid_t type,
    const void* value) {
    auto space = H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR), "create attribute space " + name);
    auto attribute = H5AttributeHandle::checked(
        H5Acreate2(
            location,
            name.c_str(),
            type,
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        "create attribute " + name);
    check_hdf5(
        H5Awrite(attribute.get(), type, value),
        "write attribute " + name);
}

void write_string_attr(
    hid_t location,
    const std::string& name,
    const std::string& value) {
    if (value.empty()) {
        throw std::invalid_argument(
            "FoF analysis string attribute is empty: " + name);
    }
    auto space = H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR), "create string attribute space " + name);
    auto type = H5TypeHandle::checked(
        H5Tcopy(H5T_C_S1), "copy string type " + name);
    check_hdf5(
        H5Tset_size(type.get(), value.size() + 1),
        "set string attribute size " + name);
    check_hdf5(
        H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
        "set string attribute padding " + name);
    auto attribute = H5AttributeHandle::checked(
        H5Acreate2(
            location,
            name.c_str(),
            type.get(),
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        "create string attribute " + name);
    check_hdf5(
        H5Awrite(attribute.get(), type.get(), value.c_str()),
        "write string attribute " + name);
}

void read_attr(
    hid_t location,
    const std::string& name,
    hid_t type,
    void* value) {
    const htri_t exists = H5Aexists(location, name.c_str());
    if (exists < 0) {
        throw std::runtime_error("Failed to query attribute " + name);
    }
    if (exists == 0) {
        throw std::runtime_error("Missing required attribute " + name);
    }
    auto attribute = H5AttributeHandle::checked(
        H5Aopen(location, name.c_str(), H5P_DEFAULT),
        "open attribute " + name);
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute.get()), "open attribute dataspace " + name);
    if (H5Sget_simple_extent_type(space.get()) != H5S_SCALAR) {
        throw std::runtime_error("Attribute must be scalar: " + name);
    }
    check_hdf5(
        H5Aread(attribute.get(), type, value),
        "read attribute " + name);
}

std::string read_string_attr(
    hid_t location,
    const std::string& name,
    std::size_t maximum_logical_bytes) {
    return read_bounded_fixed_string_attribute(
        location,
        name,
        maximum_logical_bytes,
        true,
        "FoF analysis");
}

void require_string_attr(
    hid_t location,
    const std::string& name,
    const std::string& expected,
    const char* context) {
    const std::string actual = read_string_attr(
        location, name, expected.size());
    if (actual != expected) {
        throw std::runtime_error(
            std::string(context) + " " + name + " mismatch: expected '"
            + expected + "', got '" + actual + "'");
    }
}

void require_hard_link(hid_t location, const std::string& name) {
    H5L_info_t info{};
    if (H5Lget_info(location, name.c_str(), &info, H5P_DEFAULT) < 0) {
        throw std::runtime_error("Failed to inspect HDF5 link " + name);
    }
    if (info.type != H5L_TYPE_HARD) {
        throw std::runtime_error("HDF5 object must be a hard link: " + name);
    }
}

void require_local_dataset(hid_t dataset, const std::string& name) {
    auto properties = H5PropertyHandle::checked(
        H5Dget_create_plist(dataset), "open dataset creation properties " + name);
    const int external_count = H5Pget_external_count(properties.get());
    if (external_count < 0) {
        throw std::runtime_error(
            "Failed to inspect external storage for dataset " + name);
    }
    if (external_count != 0) {
        throw std::runtime_error(
            "External HDF5 storage is forbidden for dataset " + name);
    }
    const H5D_layout_t layout = H5Pget_layout(properties.get());
    if (layout == H5D_LAYOUT_ERROR) {
        throw std::runtime_error("Failed to inspect layout for dataset " + name);
    }
    if (layout == H5D_VIRTUAL) {
        throw std::runtime_error(
            "Virtual HDF5 datasets are forbidden for " + name);
    }
}

H5DatasetHandle write_dataset_1d(
    hid_t group,
    const std::string& name,
    hid_t type,
    const void* data,
    std::size_t count,
    const std::string& unit) {
    const hsize_t dimensions[1] = {checked_hsize(count, name.c_str())};
    auto space = H5SpaceHandle::checked(
        H5Screate_simple(1, dimensions, nullptr),
        "create dataset space " + name);
    auto dataset = H5DatasetHandle::checked(
        H5Dcreate2(
            group,
            name.c_str(),
            type,
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        "create dataset " + name);
    if (count != 0) {
        check_hdf5(
            H5Dwrite(
                dataset.get(),
                type,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                data),
            "write dataset " + name);
    }
    write_string_attr(dataset.get(), schema::ATTR_UNIT, unit);
    return dataset;
}

H5DatasetHandle write_dataset_2d3(
    hid_t group,
    const std::string& name,
    hid_t type,
    const void* data,
    std::size_t rows,
    const std::string& unit) {
    const hsize_t dimensions[2] = {
        checked_hsize(rows, name.c_str()), 3};
    auto space = H5SpaceHandle::checked(
        H5Screate_simple(2, dimensions, nullptr),
        "create dataset space " + name);
    auto dataset = H5DatasetHandle::checked(
        H5Dcreate2(
            group,
            name.c_str(),
            type,
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        "create dataset " + name);
    if (rows != 0) {
        check_hdf5(
            H5Dwrite(
                dataset.get(),
                type,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                data),
            "write dataset " + name);
    }
    write_string_attr(dataset.get(), schema::ATTR_UNIT, unit);
    return dataset;
}

template <typename Value>
std::vector<Value> read_dataset_1d(
    hid_t group,
    const std::string& name,
    hid_t native_type,
    const std::string& unit) {
    require_hard_link(group, name);
    auto dataset = H5DatasetHandle::checked(
        H5Dopen2(group, name.c_str(), H5P_DEFAULT),
        "open dataset " + name);
    require_local_dataset(dataset.get(), name);
    require_string_attr(dataset.get(), schema::ATTR_UNIT, unit, "dataset");
    auto space = H5SpaceHandle::checked(
        H5Dget_space(dataset.get()), "open dataspace " + name);
    if (H5Sget_simple_extent_ndims(space.get()) != 1) {
        throw std::runtime_error("Dataset must be rank one: " + name);
    }
    hsize_t dimensions[1]{};
    if (H5Sget_simple_extent_dims(space.get(), dimensions, nullptr) < 0) {
        throw std::runtime_error("Failed to read dataset extent " + name);
    }
    if (dimensions[0] > static_cast<hsize_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Dataset extent exceeds size_t: " + name);
    }
    std::vector<Value> values(static_cast<std::size_t>(dimensions[0]));
    if (!values.empty()) {
        check_hdf5(
            H5Dread(
                dataset.get(),
                native_type,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                values.data()),
            "read dataset " + name);
    }
    return values;
}

template <typename Value>
std::vector<Value> read_dataset_2d3(
    hid_t group,
    const std::string& name,
    hid_t native_type,
    const std::string& unit,
    std::size_t& rows) {
    require_hard_link(group, name);
    auto dataset = H5DatasetHandle::checked(
        H5Dopen2(group, name.c_str(), H5P_DEFAULT),
        "open dataset " + name);
    require_local_dataset(dataset.get(), name);
    require_string_attr(dataset.get(), schema::ATTR_UNIT, unit, "dataset");
    auto space = H5SpaceHandle::checked(
        H5Dget_space(dataset.get()), "open dataspace " + name);
    if (H5Sget_simple_extent_ndims(space.get()) != 2) {
        throw std::runtime_error("Dataset must be rank two: " + name);
    }
    hsize_t dimensions[2]{};
    if (H5Sget_simple_extent_dims(space.get(), dimensions, nullptr) < 0
        || dimensions[1] != 3) {
        throw std::runtime_error("Dataset must have shape [N,3]: " + name);
    }
    if (dimensions[0] > static_cast<hsize_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Dataset row count exceeds size_t: " + name);
    }
    rows = static_cast<std::size_t>(dimensions[0]);
    if (rows > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::overflow_error("Dataset element count overflows: " + name);
    }
    std::vector<Value> values(rows * 3);
    if (!values.empty()) {
        check_hdf5(
            H5Dread(
                dataset.get(),
                native_type,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                values.data()),
            "read dataset " + name);
    }
    return values;
}

struct CatalogBindingFields {
    std::string lineage;
    std::string physics_fingerprint;
    std::string metadata_json;
    core::Real box_size{0.0};
    core::Real scale_factor{0.0};
};

void write_common_catalog_binding(
    hid_t group,
    const FoFAnalysisCatalogContext& context,
    const std::string& metadata_json,
    const std::string& lineage,
    core::Real scale_factor,
    const std::string& tracer_species,
    const std::string& selection_function,
    const std::string& normalization_convention) {
    const core::Real box_size = context.box_size;
    write_string_attr(group, schema::ATTR_PRODUCT_LINEAGE, lineage);
    write_string_attr(
        group,
        schema::ATTR_PHYSICS_FINGERPRINT,
        context.physics_fingerprint);
    write_string_attr(
        group,
        schema::ATTR_RUN_METADATA_JSON,
        metadata_json);
    write_attr(
        group,
        schema::ATTR_CATALOG_BOX_SIZE,
        H5T_NATIVE_DOUBLE,
        &box_size);
    write_attr(
        group,
        schema::ATTR_SCALE_FACTOR,
        H5T_NATIVE_DOUBLE,
        &scale_factor);
    write_string_attr(
        group,
        schema::ATTR_TRACER_SPECIES,
        tracer_species);
    write_string_attr(
        group,
        schema::ATTR_SELECTION_FUNCTION,
        selection_function);
    write_string_attr(
        group,
        schema::ATTR_COORDINATE_FRAME,
        schema::VAL_CATALOG_COORDINATE_FRAME);
    write_string_attr(
        group,
        schema::ATTR_PERIODIC_BOUNDARY_CONVENTION,
        schema::VAL_CATALOG_PERIODIC_BOUNDARY_CONVENTION);
    write_string_attr(
        group,
        schema::ATTR_EPOCH_CONVENTION,
        schema::VAL_CATALOG_EPOCH_CONVENTION);
    write_string_attr(
        group,
        schema::ATTR_PHASE_SPACE_GAUGE,
        schema::VAL_CATALOG_PHASE_SPACE_GAUGE);
    write_string_attr(
        group,
        schema::ATTR_NORMALIZATION_CONVENTION,
        normalization_convention);
    write_string_attr(
        group,
        schema::ATTR_WEIGHTING_CONVENTION,
        schema::VAL_CATALOG_WEIGHTING_CONVENTION);
}

CatalogBindingFields read_common_catalog_binding(
    hid_t group,
    const FoFAnalysisCatalogContext& context,
    const std::string& tracer_species,
    const std::string& selection_function,
    const std::string& normalization_convention,
    const char* group_label) {
    require_string_attr(
        group,
        schema::ATTR_TRACER_SPECIES,
        tracer_species,
        group_label);
    require_string_attr(
        group,
        schema::ATTR_SELECTION_FUNCTION,
        selection_function,
        group_label);
    require_string_attr(
        group,
        schema::ATTR_COORDINATE_FRAME,
        schema::VAL_CATALOG_COORDINATE_FRAME,
        group_label);
    require_string_attr(
        group,
        schema::ATTR_PERIODIC_BOUNDARY_CONVENTION,
        schema::VAL_CATALOG_PERIODIC_BOUNDARY_CONVENTION,
        group_label);
    require_string_attr(
        group,
        schema::ATTR_EPOCH_CONVENTION,
        schema::VAL_CATALOG_EPOCH_CONVENTION,
        group_label);
    require_string_attr(
        group,
        schema::ATTR_PHASE_SPACE_GAUGE,
        schema::VAL_CATALOG_PHASE_SPACE_GAUGE,
        group_label);
    require_string_attr(
        group,
        schema::ATTR_NORMALIZATION_CONVENTION,
        normalization_convention,
        group_label);
    require_string_attr(
        group,
        schema::ATTR_WEIGHTING_CONVENTION,
        schema::VAL_CATALOG_WEIGHTING_CONVENTION,
        group_label);

    const RunMetadata& expected_metadata = context.metadata;
    const std::string expected_lineage =
        checked_lineage(expected_metadata.lineage);
    const std::string expected_metadata_json = expected_metadata.to_json();
    CatalogBindingFields fields;
    fields.lineage = read_string_attr(
        group,
        schema::ATTR_PRODUCT_LINEAGE,
        expected_lineage.size());
    const ProductLineage parsed_lineage = parse_lineage(fields.lineage);
    if (parsed_lineage != expected_metadata.lineage) {
        throw std::runtime_error(
            std::string(group_label) + " lineage disagrees with context");
    }
    fields.physics_fingerprint = read_string_attr(
        group,
        schema::ATTR_PHYSICS_FINGERPRINT,
        context.physics_fingerprint.size());
    if (fields.physics_fingerprint != context.physics_fingerprint) {
        throw std::runtime_error(
            std::string(group_label) + " physics fingerprint disagrees with context");
    }
    fields.metadata_json = read_string_attr(
        group,
        schema::ATTR_RUN_METADATA_JSON,
        expected_metadata_json.size());
    if (fields.metadata_json != expected_metadata_json) {
        throw std::runtime_error(
            std::string(group_label) + " run metadata disagrees with context");
    }
    read_attr(
        group,
        schema::ATTR_CATALOG_BOX_SIZE,
        H5T_NATIVE_DOUBLE,
        &fields.box_size);
    if (!nearly_equal(fields.box_size, context.box_size)) {
        throw std::runtime_error(
            std::string(group_label) + " box size disagrees with context");
    }
    read_attr(
        group,
        schema::ATTR_SCALE_FACTOR,
        H5T_NATIVE_DOUBLE,
        &fields.scale_factor);
    if (!std::isfinite(fields.scale_factor) || fields.scale_factor <= 0.0) {
        throw std::runtime_error(
            std::string(group_label) + " scale factor is invalid");
    }
    if (context.expected_scale_factor.has_value()
        && !nearly_equal(
            fields.scale_factor, *context.expected_scale_factor)) {
        throw std::runtime_error(
            std::string(group_label) + " scale factor disagrees with context");
    }
    return fields;
}

void validate_property_row(
    const analysis::HaloDerivedProperties& property,
    core::Real box_size,
    core::Real scale_factor) {
    if (property.particle_count == 0
        || !std::isfinite(property.mass)
        || property.mass <= 0.0
        || !finite_vec3(property.center_of_mass)
        || !finite_vec3(property.mean_momentum)
        || !finite_vec3(property.mean_velocity)
        || !finite_vec3(property.velocity_dispersion)
        || !std::isfinite(property.velocity_dispersion_3d)
        || !finite_vec3(property.angular_momentum_comoving)
        || !finite_vec3(property.specific_angular_momentum_comoving)
        || !std::isfinite(property.rms_radius_comoving)
        || !std::isfinite(property.max_radius_comoving)) {
        throw std::invalid_argument(
            "Halo derived-property row contains invalid or non-finite fields");
    }
    if (property.center_of_mass.x < 0.0
        || property.center_of_mass.y < 0.0
        || property.center_of_mass.z < 0.0
        || property.center_of_mass.x >= box_size
        || property.center_of_mass.y >= box_size
        || property.center_of_mass.z >= box_size) {
        throw std::invalid_argument(
            "Halo derived-property center lies outside periodic [0,L) box");
    }
    if (property.velocity_dispersion.x < 0.0
        || property.velocity_dispersion.y < 0.0
        || property.velocity_dispersion.z < 0.0
        || property.velocity_dispersion_3d < 0.0
        || property.rms_radius_comoving < 0.0
        || property.max_radius_comoving < 0.0
        || property.rms_radius_comoving
            > property.max_radius_comoving
                + 512.0 * std::numeric_limits<core::Real>::epsilon()
                    * property.max_radius_comoving) {
        throw std::invalid_argument(
            "Halo derived-property dispersions or radii are inconsistent");
    }
    if (!math::matches_independently_rounded_quotient_with_exact_denominator(
            property.mean_momentum.x,
            scale_factor,
            property.mean_velocity.x)
        || !math::matches_independently_rounded_quotient_with_exact_denominator(
            property.mean_momentum.y,
            scale_factor,
            property.mean_velocity.y)
        || !math::matches_independently_rounded_quotient_with_exact_denominator(
            property.mean_momentum.z,
            scale_factor,
            property.mean_velocity.z)) {
        throw std::invalid_argument(
            "Halo mean velocity is inconsistent with canonical momentum and scale factor");
    }
    if (!math::matches_independently_rounded_quotient(
            property.angular_momentum_comoving.x,
            property.mass,
            property.specific_angular_momentum_comoving.x)
        || !math::matches_independently_rounded_quotient(
            property.angular_momentum_comoving.y,
            property.mass,
            property.specific_angular_momentum_comoving.y)
        || !math::matches_independently_rounded_quotient(
            property.angular_momentum_comoving.z,
            property.mass,
            property.specific_angular_momentum_comoving.z)) {
        throw std::invalid_argument(
            "Halo specific angular momentum is inconsistent with total mass");
    }
    const core::Real expected_dispersion_3d = core::scale_safe_norm3(
        property.velocity_dispersion.x,
        property.velocity_dispersion.y,
        property.velocity_dispersion.z);
    if (!nearly_equal(
            property.velocity_dispersion_3d,
            expected_dispersion_3d)) {
        throw std::invalid_argument(
            "Halo three-dimensional velocity dispersion is inconsistent");
    }
}

void require_matching_catalog_binding(
    const CatalogBindingFields& memberships,
    const CatalogBindingFields& properties) {
    if (memberships.lineage != properties.lineage
        || memberships.physics_fingerprint != properties.physics_fingerprint
        || memberships.metadata_json != properties.metadata_json
        || !nearly_equal(memberships.box_size, properties.box_size)
        || !nearly_equal(memberships.scale_factor, properties.scale_factor)) {
        throw std::runtime_error(
            "FoF membership and derived-property metadata disagree");
    }
}

} // namespace

FoFAnalysisCatalogIO::FoFAnalysisCatalogIO(
    FoFAnalysisCatalogContext context)
    : context_(std::move(context)) {
    if (!std::isfinite(context_.box_size) || context_.box_size <= 0.0) {
        throw std::invalid_argument(
            "FoF analysis catalog context requires a finite positive box size");
    }
    if (context_.physics_fingerprint.empty()) {
        throw std::invalid_argument(
            "FoF analysis catalog context requires a physics fingerprint");
    }
    if (context_.metadata.lineage == ProductLineage::Unknown) {
        throw std::invalid_argument(
            "FoF analysis catalog context requires explicit metadata lineage");
    }
    if (context_.expected_scale_factor.has_value()) {
        const core::Real scale_factor = *context_.expected_scale_factor;
        if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
            throw std::invalid_argument(
                "FoF analysis catalog context requires a finite positive epoch");
        }
    }
}

void FoFAnalysisCatalogIO::write(
    const std::string& filename,
    std::span<const halo::FoFMembership> memberships,
    std::span<const analysis::HaloDerivedProperties> properties,
    const core::ParticleStore& particles,
    core::Real linking_length_b,
    std::size_t min_particles,
    core::Real scale_factor,
    ProductLineage lineage) const {
    if (filename.empty()) {
        throw std::invalid_argument(
            "Split FoF analysis catalog filename must not be empty");
    }
    if (!std::isfinite(linking_length_b) || linking_length_b <= 0.0) {
        throw std::invalid_argument(
            "FoF linking length b must be finite and positive");
    }
    if (min_particles == 0) {
        throw std::invalid_argument("FoF min_particles must be positive");
    }
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        throw std::invalid_argument(
            "FoF analysis scale factor must be finite and positive");
    }
    if (context_.expected_scale_factor.has_value()
        && !nearly_equal(scale_factor, *context_.expected_scale_factor)) {
        throw std::invalid_argument(
            "FoF analysis scale factor disagrees with its context");
    }
    if (memberships.size() != properties.size()) {
        throw std::invalid_argument(
            "FoF membership and derived-property row counts differ");
    }
    const std::string lineage_string = checked_lineage(lineage);
    const RunMetadata& metadata = context_.metadata;
    if (lineage != metadata.lineage) {
        throw std::invalid_argument(
            "FoF analysis catalog lineage disagrees with its context");
    }
    const std::string metadata_json = metadata.to_json();

    const std::size_t owned = particles.num_owned_particles();
    if (particles.get_ids().size() < owned) {
        throw std::logic_error(
            "Particle ID storage is shorter than the owned count");
    }
    const auto particle_ids = particles.get_ids().first(owned);
    core::require_unique_particle_ids_bounded(
        particle_ids, "Split FoF analysis source particle IDs");
    const core::Real box_size = context_.box_size;

    const std::size_t group_count = memberships.size();
    if (group_count > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::overflow_error("FoF analysis vector fields overflow size_t");
    }
    std::vector<std::uint64_t> group_ids(group_count);
    std::vector<std::uint64_t> group_sizes(group_count);
    std::vector<std::uint64_t> group_offsets(group_count + 1, 0);
    std::vector<std::uint64_t> member_ids;
    std::vector<std::uint64_t> property_ids(group_count);
    std::vector<std::uint64_t> property_counts(group_count);
    std::vector<core::Real> masses(group_count);
    std::vector<core::Real> centers(group_count * 3);
    std::vector<core::Real> mean_momenta(group_count * 3);
    std::vector<core::Real> mean_velocities(group_count * 3);
    std::vector<core::Real> dispersions(group_count * 3);
    std::vector<core::Real> dispersion_3d(group_count);
    std::vector<core::Real> angular_momenta(group_count * 3);
    std::vector<core::Real> specific_angular_momenta(group_count * 3);
    std::vector<core::Real> rms_radii(group_count);
    std::vector<core::Real> max_radii(group_count);

    for (std::size_t index = 0; index < group_count; ++index) {
        const auto& membership = memberships[index];
        const auto& property = properties[index];
        if (membership.id != index || property.halo_id != index
            || property.halo_id != membership.id) {
            throw std::invalid_argument(
                "FoF memberships and properties require contiguous matching halo IDs");
        }
        if (membership.particle_indices.size() < min_particles
            || membership.particle_indices.size() != property.particle_count) {
            throw std::invalid_argument(
                "FoF membership size disagrees with threshold or property count");
        }
        validate_property_row(property, box_size, scale_factor);

        group_ids[index] = checked_u64(membership.id, "FoF group ID");
        group_sizes[index] = checked_u64(
            membership.particle_indices.size(), "FoF group size");
        group_offsets[index] = checked_u64(
            member_ids.size(), "FoF group offset");

        math::ExactPositiveDoubleSum recomputed_mass;
        core::ParticleId previous_id = 0;
        bool first_member = true;
        for (const std::size_t particle_index : membership.particle_indices) {
            if (particle_index >= owned) {
                throw std::out_of_range(
                    "FoF member index exceeds the owned particle count");
            }
            const core::ParticleId particle_id = particle_ids[particle_index];
            if (!first_member && particle_id <= previous_id) {
                throw std::invalid_argument(
                    "FoF member ParticleIDs must be strictly increasing within each group");
            }
            first_member = false;
            previous_id = particle_id;
            const core::Real mass = particles.mass_at(particle_index);
            if (!std::isfinite(mass) || mass <= 0.0) {
                throw std::invalid_argument(
                    "FoF member mass must be finite and positive");
            }
            recomputed_mass.add(mass);
            member_ids.push_back(particle_id);
        }
        const core::Real recomputed_mass_real = recomputed_mass.value();
        if (!std::isfinite(recomputed_mass_real) || recomputed_mass_real <= 0.0
            || recomputed_mass_real != property.mass) {
            throw std::invalid_argument(
                "Derived halo mass does not equal its member particle mass sum");
        }

        property_ids[index] = checked_u64(property.halo_id, "property halo ID");
        property_counts[index] = checked_u64(
            property.particle_count, "property particle count");
        masses[index] = property.mass;
        const std::array<core::Vec3, 6> vectors{
            property.center_of_mass,
            property.mean_momentum,
            property.mean_velocity,
            property.velocity_dispersion,
            property.angular_momentum_comoving,
            property.specific_angular_momentum_comoving};
        std::array<std::vector<core::Real>*, 6> destinations{
            &centers,
            &mean_momenta,
            &mean_velocities,
            &dispersions,
            &angular_momenta,
            &specific_angular_momenta};
        for (std::size_t vector_index = 0;
             vector_index < vectors.size();
             ++vector_index) {
            (*destinations[vector_index])[3 * index + 0] = vectors[vector_index].x;
            (*destinations[vector_index])[3 * index + 1] = vectors[vector_index].y;
            (*destinations[vector_index])[3 * index + 2] = vectors[vector_index].z;
        }
        dispersion_3d[index] = property.velocity_dispersion_3d;
        rms_radii[index] = property.rms_radius_comoving;
        max_radii[index] = property.max_radius_comoving;
    }
    core::require_unique_particle_ids_bounded(
        member_ids, "Split FoF analysis member IDs across groups");
    group_offsets[group_count] = checked_u64(
        member_ids.size(), "FoF final group offset");

    const std::filesystem::path final_path(filename);
    const std::filesystem::path temporary_path = final_path.string() + ".tmp";
    std::error_code error;
    std::filesystem::remove(temporary_path, error);
    try {
        auto file = H5FileHandle::checked(
            H5Fcreate(
                temporary_path.string().c_str(),
                H5F_ACC_TRUNC,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "create split FoF analysis catalog");
        auto group_creation_properties =
            provenance_group_creation_properties(
                "create split FoF provenance groups");
        auto membership_group = H5GroupHandle::checked(
            H5Gcreate2(
                file.get(),
                schema::GROUP_FOF_MEMBERSHIP.c_str(),
                H5P_DEFAULT,
                group_creation_properties.get(),
                H5P_DEFAULT),
            "create FoF membership group");
        auto property_group = H5GroupHandle::checked(
            H5Gcreate2(
                file.get(),
                schema::GROUP_HALO_DERIVED_PROPERTIES.c_str(),
                H5P_DEFAULT,
                group_creation_properties.get(),
                H5P_DEFAULT),
            "create halo derived-property group");

        const unsigned long long min_particles_disk =
            static_cast<unsigned long long>(min_particles);
        write_attr(
            membership_group.get(),
            schema::ATTR_LINKING_LENGTH_B,
            H5T_NATIVE_DOUBLE,
            &linking_length_b);
        write_attr(
            membership_group.get(),
            schema::ATTR_MIN_PARTICLES,
            H5T_NATIVE_ULLONG,
            &min_particles_disk);
        write_common_catalog_binding(
            membership_group.get(),
            context_,
            metadata_json,
            lineage_string,
            scale_factor,
            schema::VAL_FOF_MEMBERSHIP_TRACER_SPECIES,
            schema::VAL_FOF_MEMBERSHIP_SELECTION_FUNCTION,
            schema::VAL_FOF_MEMBERSHIP_NORMALIZATION_CONVENTION);

        write_string_attr(
            property_group.get(),
            schema::ATTR_MEMBERSHIP_GROUP_PATH,
            schema::VAL_FOF_MEMBERSHIP_GROUP_PATH);
        write_string_attr(
            property_group.get(),
            schema::ATTR_MASS_DEFINITION,
            schema::VAL_FOF_MASS_DEFINITION);
        write_common_catalog_binding(
            property_group.get(),
            context_,
            metadata_json,
            lineage_string,
            scale_factor,
            schema::VAL_HALO_DERIVED_TRACER_SPECIES,
            schema::VAL_HALO_DERIVED_SELECTION_FUNCTION,
            schema::VAL_HALO_DERIVED_NORMALIZATION_CONVENTION);

        const hid_t real_type = sizeof(core::Real) == sizeof(double)
            ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
        write_dataset_1d(
            membership_group.get(),
            schema::DATASET_MEMBERSHIP_GROUP_IDS,
            H5T_NATIVE_UINT64,
            group_ids.data(),
            group_ids.size(),
            schema::UNIT_DIMENSIONLESS);
        write_dataset_1d(
            membership_group.get(),
            schema::DATASET_MEMBERSHIP_GROUP_SIZES,
            H5T_NATIVE_UINT64,
            group_sizes.data(),
            group_sizes.size(),
            schema::UNIT_PARTICLES);
        write_dataset_1d(
            membership_group.get(),
            schema::DATASET_MEMBERSHIP_GROUP_OFFSETS,
            H5T_NATIVE_UINT64,
            group_offsets.data(),
            group_offsets.size(),
            schema::UNIT_PARTICLES);
        write_dataset_1d(
            membership_group.get(),
            schema::DATASET_MEMBERSHIP_PARTICLE_IDS,
            H5T_NATIVE_UINT64,
            member_ids.data(),
            member_ids.size(),
            schema::UNIT_DIMENSIONLESS);

        write_dataset_1d(
            property_group.get(),
            schema::DATASET_DERIVED_HALO_IDS,
            H5T_NATIVE_UINT64,
            property_ids.data(),
            property_ids.size(),
            schema::UNIT_DIMENSIONLESS);
        write_dataset_1d(
            property_group.get(),
            schema::DATASET_DERIVED_PARTICLE_COUNTS,
            H5T_NATIVE_UINT64,
            property_counts.data(),
            property_counts.size(),
            schema::UNIT_PARTICLES);
        write_dataset_1d(
            property_group.get(),
            schema::DATASET_DERIVED_MASSES,
            real_type,
            masses.data(),
            masses.size(),
            schema::UNIT_CODE_MASS);
        write_dataset_2d3(
            property_group.get(),
            schema::DATASET_DERIVED_CENTERS,
            real_type,
            centers.data(),
            group_count,
            schema::UNIT_COMOVING_MPC_PER_H);
        auto momentum_dataset = write_dataset_2d3(
            property_group.get(),
            schema::DATASET_DERIVED_MEAN_MOMENTA,
            real_type,
            mean_momenta.data(),
            group_count,
            schema::UNIT_CANONICAL_MOMENTUM);
        write_string_attr(
            momentum_dataset.get(),
            schema::ATTR_MOMENTUM_CONVENTION,
            schema::VAL_MOMENTUM_CONVENTION);
        auto velocity_dataset = write_dataset_2d3(
            property_group.get(),
            schema::DATASET_DERIVED_MEAN_VELOCITIES,
            real_type,
            mean_velocities.data(),
            group_count,
            schema::UNIT_PECULIAR_VELOCITY);
        write_string_attr(
            velocity_dataset.get(),
            schema::ATTR_VELOCITY_CONVENTION,
            schema::VAL_FOF_VELOCITY_CONVENTION);
        write_dataset_2d3(
            property_group.get(),
            schema::DATASET_DERIVED_VELOCITY_DISPERSIONS,
            real_type,
            dispersions.data(),
            group_count,
            schema::UNIT_PECULIAR_VELOCITY);
        write_dataset_1d(
            property_group.get(),
            schema::DATASET_DERIVED_VELOCITY_DISPERSION_3D,
            real_type,
            dispersion_3d.data(),
            dispersion_3d.size(),
            schema::UNIT_PECULIAR_VELOCITY);
        auto angular_dataset = write_dataset_2d3(
            property_group.get(),
            schema::DATASET_DERIVED_ANGULAR_MOMENTA_COMOVING,
            real_type,
            angular_momenta.data(),
            group_count,
            schema::UNIT_COMOVING_ANGULAR_MOMENTUM);
        write_string_attr(
            angular_dataset.get(),
            "AngularMomentumConvention",
            schema::VAL_ANGULAR_MOMENTUM_CONVENTION);
        auto specific_angular_dataset = write_dataset_2d3(
            property_group.get(),
            schema::DATASET_DERIVED_SPECIFIC_ANGULAR_MOMENTA_COMOVING,
            real_type,
            specific_angular_momenta.data(),
            group_count,
            schema::UNIT_SPECIFIC_COMOVING_ANGULAR_MOMENTUM);
        write_string_attr(
            specific_angular_dataset.get(),
            "AngularMomentumConvention",
            schema::VAL_ANGULAR_MOMENTUM_CONVENTION);
        write_dataset_1d(
            property_group.get(),
            schema::DATASET_DERIVED_RMS_RADII_COMOVING,
            real_type,
            rms_radii.data(),
            rms_radii.size(),
            schema::UNIT_COMOVING_MPC_PER_H);
        write_dataset_1d(
            property_group.get(),
            schema::DATASET_DERIVED_MAX_RADII_COMOVING,
            real_type,
            max_radii.data(),
            max_radii.size(),
            schema::UNIT_COMOVING_MPC_PER_H);

        check_hdf5(
            H5Fflush(file.get(), H5F_SCOPE_GLOBAL),
            "flush split FoF analysis catalog");
    } catch (...) {
        std::filesystem::remove(temporary_path, error);
        throw;
    }

    std::filesystem::rename(temporary_path, final_path, error);
    if (error) {
        std::filesystem::remove(temporary_path);
        throw std::runtime_error(
            "Atomic split FoF analysis catalog rename failed: "
            + error.message());
    }
}

FoFAnalysisCatalog FoFAnalysisCatalogIO::read(
    const std::string& filename) const {
    if (filename.empty()) {
        throw std::invalid_argument(
            "Split FoF analysis catalog filename must not be empty");
    }
    auto file = H5FileHandle::checked(
        H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
        "open split FoF analysis catalog");
    for (const std::string* group_name : {
             &schema::GROUP_FOF_MEMBERSHIP,
             &schema::GROUP_HALO_DERIVED_PROPERTIES}) {
        const htri_t exists = H5Lexists(
            file.get(), group_name->c_str(), H5P_DEFAULT);
        if (exists <= 0) {
            throw std::runtime_error(
                "Missing required split FoF group " + *group_name);
        }
        require_hard_link(file.get(), *group_name);
    }
    auto membership_group = H5GroupHandle::checked(
        H5Gopen2(
            file.get(),
            schema::GROUP_FOF_MEMBERSHIP.c_str(),
            H5P_DEFAULT),
        "open FoF membership group");
    auto property_group = H5GroupHandle::checked(
        H5Gopen2(
            file.get(),
            schema::GROUP_HALO_DERIVED_PROPERTIES.c_str(),
            H5P_DEFAULT),
        "open halo derived-property group");

    FoFAnalysisCatalog catalog;
    read_attr(
        membership_group.get(),
        schema::ATTR_LINKING_LENGTH_B,
        H5T_NATIVE_DOUBLE,
        &catalog.memberships.linking_length_b);
    if (!std::isfinite(catalog.memberships.linking_length_b)
        || catalog.memberships.linking_length_b <= 0.0) {
        throw std::runtime_error("FoF membership linking length is invalid");
    }
    unsigned long long min_particles_disk = 0;
    read_attr(
        membership_group.get(),
        schema::ATTR_MIN_PARTICLES,
        H5T_NATIVE_ULLONG,
        &min_particles_disk);
    if (min_particles_disk == 0
        || min_particles_disk
            > static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("FoF membership min_particles is invalid");
    }
    catalog.memberships.min_particles =
        static_cast<std::size_t>(min_particles_disk);
    const CatalogBindingFields membership_binding =
        read_common_catalog_binding(
            membership_group.get(),
            context_,
            schema::VAL_FOF_MEMBERSHIP_TRACER_SPECIES,
            schema::VAL_FOF_MEMBERSHIP_SELECTION_FUNCTION,
            schema::VAL_FOF_MEMBERSHIP_NORMALIZATION_CONVENTION,
            "FoF membership catalog");
    catalog.memberships.lineage_string = membership_binding.lineage;
    catalog.memberships.lineage = parse_lineage(membership_binding.lineage);
    catalog.memberships.scale_factor = membership_binding.scale_factor;
    catalog.memberships.run_metadata_json =
        membership_binding.metadata_json;

    catalog.properties.membership_group_path = read_string_attr(
        property_group.get(),
        schema::ATTR_MEMBERSHIP_GROUP_PATH,
        schema::VAL_FOF_MEMBERSHIP_GROUP_PATH.size());
    if (catalog.properties.membership_group_path
        != schema::VAL_FOF_MEMBERSHIP_GROUP_PATH) {
        throw std::runtime_error(
            "Halo properties reference an unsupported membership group path");
    }
    catalog.properties.mass_definition = read_string_attr(
        property_group.get(),
        schema::ATTR_MASS_DEFINITION,
        schema::VAL_FOF_MASS_DEFINITION.size());
    if (catalog.properties.mass_definition
        != schema::VAL_FOF_MASS_DEFINITION) {
        throw std::runtime_error(
            "Unsupported halo derived-property mass definition");
    }
    const CatalogBindingFields property_binding =
        read_common_catalog_binding(
            property_group.get(),
            context_,
            schema::VAL_HALO_DERIVED_TRACER_SPECIES,
            schema::VAL_HALO_DERIVED_SELECTION_FUNCTION,
            schema::VAL_HALO_DERIVED_NORMALIZATION_CONVENTION,
            "Halo derived-property catalog");
    require_matching_catalog_binding(
        membership_binding, property_binding);
    catalog.properties.lineage_string = property_binding.lineage;
    catalog.properties.lineage = parse_lineage(property_binding.lineage);
    catalog.properties.scale_factor = property_binding.scale_factor;
    catalog.properties.run_metadata_json = property_binding.metadata_json;

    const auto group_ids = read_dataset_1d<std::uint64_t>(
        membership_group.get(),
        schema::DATASET_MEMBERSHIP_GROUP_IDS,
        H5T_NATIVE_UINT64,
        schema::UNIT_DIMENSIONLESS);
    const auto group_sizes = read_dataset_1d<std::uint64_t>(
        membership_group.get(),
        schema::DATASET_MEMBERSHIP_GROUP_SIZES,
        H5T_NATIVE_UINT64,
        schema::UNIT_PARTICLES);
    const auto group_offsets = read_dataset_1d<std::uint64_t>(
        membership_group.get(),
        schema::DATASET_MEMBERSHIP_GROUP_OFFSETS,
        H5T_NATIVE_UINT64,
        schema::UNIT_PARTICLES);
    require_hard_link(
        membership_group.get(), schema::DATASET_MEMBERSHIP_PARTICLE_IDS);
    auto member_dataset = H5DatasetHandle::checked(
        H5Dopen2(
            membership_group.get(),
            schema::DATASET_MEMBERSHIP_PARTICLE_IDS.c_str(),
            H5P_DEFAULT),
        "open FoF member ParticleID dataset");
    require_local_dataset(
        member_dataset.get(), schema::DATASET_MEMBERSHIP_PARTICLE_IDS);
    require_string_attr(
        member_dataset.get(),
        schema::ATTR_UNIT,
        schema::UNIT_DIMENSIONLESS,
        "dataset");
    auto member_file_space = H5SpaceHandle::checked(
        H5Dget_space(member_dataset.get()),
        "open FoF member ParticleID dataspace");
    if (H5Sget_simple_extent_ndims(member_file_space.get()) != 1) {
        throw std::runtime_error(
            "FoF member ParticleID dataset must be rank one");
    }
    hsize_t member_dimensions[1]{};
    if (H5Sget_simple_extent_dims(
            member_file_space.get(), member_dimensions, nullptr) < 0
        || member_dimensions[0] > static_cast<hsize_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "FoF member ParticleID dataset extent is invalid");
    }
    const std::size_t member_count =
        static_cast<std::size_t>(member_dimensions[0]);
    if (group_sizes.size() != group_ids.size()
        || group_offsets.size() != group_ids.size() + 1
        || group_offsets.empty()
        || group_offsets.front() != 0
        || group_offsets.back() != member_count) {
        throw std::runtime_error(
            "FoF membership dataset extents or terminal offset are inconsistent");
    }

    catalog.memberships.groups.resize(group_ids.size());
    for (std::size_t index = 0; index < group_ids.size(); ++index) {
        if (group_ids[index] != index
            || group_sizes[index] < catalog.memberships.min_particles
            || group_offsets[index] > group_offsets[index + 1]
            || group_offsets[index + 1] - group_offsets[index]
                != group_sizes[index]) {
            throw std::runtime_error(
                "FoF membership IDs, sizes, or offsets are inconsistent");
        }
        auto& group = catalog.memberships.groups[index];
        group.id = index;
        const std::size_t begin = checked_size(
            group_offsets[index], "FoF membership begin offset");
        const std::size_t end = checked_size(
            group_offsets[index + 1], "FoF membership end offset");
        const std::size_t group_size = end - begin;
        group.member_ids.resize(group_size);
        if (group_size != 0) {
            const hsize_t start[1] = {checked_hsize(
                begin, "FoF member ParticleID hyperslab offset")};
            const hsize_t count[1] = {checked_hsize(
                group_size, "FoF member ParticleID hyperslab size")};
            check_hdf5(
                H5Sselect_hyperslab(
                    member_file_space.get(),
                    H5S_SELECT_SET,
                    start,
                    nullptr,
                    count,
                    nullptr),
                "select FoF member ParticleID hyperslab");
            auto memory_space = H5SpaceHandle::checked(
                H5Screate_simple(1, count, nullptr),
                "create FoF member ParticleID memory space");
            check_hdf5(
                H5Dread(
                    member_dataset.get(),
                    H5T_NATIVE_UINT64,
                    memory_space.get(),
                    member_file_space.get(),
                    H5P_DEFAULT,
                    group.member_ids.data()),
                "read FoF member ParticleID hyperslab");
        }
        for (std::size_t member = 0; member < group.member_ids.size(); ++member) {
            if (member != 0
                && group.member_ids[member] <= group.member_ids[member - 1]) {
                throw std::runtime_error(
                    "FoF member IDs are not strictly increasing within a group");
            }
        }
    }

    struct MemberCursor {
        core::ParticleId id{0};
        std::size_t group{0};
        std::size_t offset{0};
    };
    const auto later_member = [](const MemberCursor& lhs, const MemberCursor& rhs) {
        return lhs.id > rhs.id;
    };
    std::priority_queue<
        MemberCursor,
        std::vector<MemberCursor>,
        decltype(later_member)> member_heap(later_member);
    for (std::size_t group_index = 0;
         group_index < catalog.memberships.groups.size();
         ++group_index) {
        const auto& ids = catalog.memberships.groups[group_index].member_ids;
        if (!ids.empty()) member_heap.push({ids.front(), group_index, 0});
    }
    core::ParticleId previous_member_id = 0;
    bool first_global_member = true;
    while (!member_heap.empty()) {
        const MemberCursor cursor = member_heap.top();
        member_heap.pop();
        if (!first_global_member && cursor.id == previous_member_id) {
            throw std::runtime_error(
                "FoF member IDs are not globally unique across groups");
        }
        first_global_member = false;
        previous_member_id = cursor.id;
        const auto& ids = catalog.memberships.groups[cursor.group].member_ids;
        const std::size_t next = cursor.offset + 1;
        if (next < ids.size()) {
            member_heap.push({ids[next], cursor.group, next});
        }
    }

    const auto property_ids = read_dataset_1d<std::uint64_t>(
        property_group.get(),
        schema::DATASET_DERIVED_HALO_IDS,
        H5T_NATIVE_UINT64,
        schema::UNIT_DIMENSIONLESS);
    const auto property_counts = read_dataset_1d<std::uint64_t>(
        property_group.get(),
        schema::DATASET_DERIVED_PARTICLE_COUNTS,
        H5T_NATIVE_UINT64,
        schema::UNIT_PARTICLES);
    const hid_t real_type = sizeof(core::Real) == sizeof(double)
        ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    const auto masses = read_dataset_1d<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_MASSES,
        real_type,
        schema::UNIT_CODE_MASS);
    std::size_t center_rows = 0;
    const auto centers = read_dataset_2d3<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_CENTERS,
        real_type,
        schema::UNIT_COMOVING_MPC_PER_H,
        center_rows);
    std::size_t momentum_rows = 0;
    const auto mean_momenta = read_dataset_2d3<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_MEAN_MOMENTA,
        real_type,
        schema::UNIT_CANONICAL_MOMENTUM,
        momentum_rows);
    {
        auto dataset = H5DatasetHandle::checked(
            H5Dopen2(
                property_group.get(),
                schema::DATASET_DERIVED_MEAN_MOMENTA.c_str(),
                H5P_DEFAULT),
            "open mean momentum convention");
        require_string_attr(
            dataset.get(),
            schema::ATTR_MOMENTUM_CONVENTION,
            schema::VAL_MOMENTUM_CONVENTION,
            "mean momentum dataset");
    }
    std::size_t velocity_rows = 0;
    const auto mean_velocities = read_dataset_2d3<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_MEAN_VELOCITIES,
        real_type,
        schema::UNIT_PECULIAR_VELOCITY,
        velocity_rows);
    {
        auto dataset = H5DatasetHandle::checked(
            H5Dopen2(
                property_group.get(),
                schema::DATASET_DERIVED_MEAN_VELOCITIES.c_str(),
                H5P_DEFAULT),
            "open mean velocity convention");
        require_string_attr(
            dataset.get(),
            schema::ATTR_VELOCITY_CONVENTION,
            schema::VAL_FOF_VELOCITY_CONVENTION,
            "mean velocity dataset");
    }
    std::size_t dispersion_rows = 0;
    const auto dispersions = read_dataset_2d3<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_VELOCITY_DISPERSIONS,
        real_type,
        schema::UNIT_PECULIAR_VELOCITY,
        dispersion_rows);
    const auto dispersion_3d = read_dataset_1d<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_VELOCITY_DISPERSION_3D,
        real_type,
        schema::UNIT_PECULIAR_VELOCITY);
    std::size_t angular_rows = 0;
    const auto angular_momenta = read_dataset_2d3<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_ANGULAR_MOMENTA_COMOVING,
        real_type,
        schema::UNIT_COMOVING_ANGULAR_MOMENTUM,
        angular_rows);
    {
        auto dataset = H5DatasetHandle::checked(
            H5Dopen2(
                property_group.get(),
                schema::DATASET_DERIVED_ANGULAR_MOMENTA_COMOVING.c_str(),
                H5P_DEFAULT),
            "open angular momentum convention");
        require_string_attr(
            dataset.get(),
            "AngularMomentumConvention",
            schema::VAL_ANGULAR_MOMENTUM_CONVENTION,
            "angular momentum dataset");
    }
    std::size_t specific_angular_rows = 0;
    const auto specific_angular_momenta = read_dataset_2d3<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_SPECIFIC_ANGULAR_MOMENTA_COMOVING,
        real_type,
        schema::UNIT_SPECIFIC_COMOVING_ANGULAR_MOMENTUM,
        specific_angular_rows);
    {
        auto dataset = H5DatasetHandle::checked(
            H5Dopen2(
                property_group.get(),
                schema::DATASET_DERIVED_SPECIFIC_ANGULAR_MOMENTA_COMOVING.c_str(),
                H5P_DEFAULT),
            "open specific angular momentum convention");
        require_string_attr(
            dataset.get(),
            "AngularMomentumConvention",
            schema::VAL_ANGULAR_MOMENTUM_CONVENTION,
            "specific angular momentum dataset");
    }
    const auto rms_radii = read_dataset_1d<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_RMS_RADII_COMOVING,
        real_type,
        schema::UNIT_COMOVING_MPC_PER_H);
    const auto max_radii = read_dataset_1d<core::Real>(
        property_group.get(),
        schema::DATASET_DERIVED_MAX_RADII_COMOVING,
        real_type,
        schema::UNIT_COMOVING_MPC_PER_H);

    const std::size_t property_count = property_ids.size();
    if (property_count != catalog.memberships.groups.size()
        || property_counts.size() != property_count
        || masses.size() != property_count
        || center_rows != property_count
        || momentum_rows != property_count
        || velocity_rows != property_count
        || dispersion_rows != property_count
        || dispersion_3d.size() != property_count
        || angular_rows != property_count
        || specific_angular_rows != property_count
        || rms_radii.size() != property_count
        || max_radii.size() != property_count) {
        throw std::runtime_error(
            "Halo derived-property dataset extents disagree with memberships");
    }

    catalog.properties.halos.resize(property_count);
    for (std::size_t index = 0; index < property_count; ++index) {
        if (property_ids[index] != index
            || property_counts[index]
                != catalog.memberships.groups[index].member_ids.size()) {
            throw std::runtime_error(
                "Halo property IDs or particle counts disagree with memberships");
        }
        auto& property = catalog.properties.halos[index];
        property.halo_id = index;
        property.particle_count = checked_size(
            property_counts[index], "halo property particle count");
        property.mass = masses[index];
        property.center_of_mass = {
            centers[3 * index + 0],
            centers[3 * index + 1],
            centers[3 * index + 2]};
        property.mean_momentum = {
            mean_momenta[3 * index + 0],
            mean_momenta[3 * index + 1],
            mean_momenta[3 * index + 2]};
        property.mean_velocity = {
            mean_velocities[3 * index + 0],
            mean_velocities[3 * index + 1],
            mean_velocities[3 * index + 2]};
        property.velocity_dispersion = {
            dispersions[3 * index + 0],
            dispersions[3 * index + 1],
            dispersions[3 * index + 2]};
        property.velocity_dispersion_3d = dispersion_3d[index];
        property.angular_momentum_comoving = {
            angular_momenta[3 * index + 0],
            angular_momenta[3 * index + 1],
            angular_momenta[3 * index + 2]};
        property.specific_angular_momentum_comoving = {
            specific_angular_momenta[3 * index + 0],
            specific_angular_momenta[3 * index + 1],
            specific_angular_momenta[3 * index + 2]};
        property.rms_radius_comoving = rms_radii[index];
        property.max_radius_comoving = max_radii[index];
        validate_property_row(
            property,
            property_binding.box_size,
            property_binding.scale_factor);
    }
    return catalog;
}

} // namespace cosmo_nbody::io
