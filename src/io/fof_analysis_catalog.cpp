#include "cosmo_nbody/io/fof_analysis_catalog.hpp"

#include "cosmo_nbody/core/id_uniqueness.hpp"
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

hsize_t checked_hsize(std::size_t value, const char* label) {
    if constexpr (sizeof(std::size_t) > sizeof(hsize_t)) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<hsize_t>::max())) {
            throw std::overflow_error(std::string(label) + " exceeds hsize_t");
        }
    }
    return static_cast<hsize_t>(value);
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

} // namespace cosmo_nbody::io
