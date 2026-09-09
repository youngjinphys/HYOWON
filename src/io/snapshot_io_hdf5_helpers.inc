#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/io/bounded_hdf5_string.hpp"
#include "cosmo_nbody/io/metadata.hpp"
#include "cosmo_nbody/io/output_schema.hpp"
#include "cosmo_nbody/io/hdf5_handle.hpp"
#include "cosmo_nbody/io/snapshot_physics.hpp"
#include "cosmo_nbody/io/verified_snapshot_source.hpp"
#include "snapshot_hdf5_file_type.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cosmo_nbody {
namespace io {

namespace {

bool snapshot_metadata_nearly_equal(double lhs, double rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) return false;
    if (lhs == rhs) return true;
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    if (scale == 0.0) return true;
    return std::abs(lhs - rhs)
        <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool attribute_exists(hid_t location, const std::string& name) {
    const htri_t exists = H5Aexists(location, name.c_str());
    if (exists < 0) {
        throw std::runtime_error(
            "Failed to query HDF5 attribute existence: " + name);
    }
    return exists > 0;
}

bool link_exists(hid_t location, const std::string& name) {
    const htri_t exists = H5Lexists(location, name.c_str(), H5P_DEFAULT);
    if (exists < 0) {
        throw std::runtime_error(
            "Failed to query HDF5 link existence: " + name);
    }
    return exists > 0;
}

void require_hard_link(
    hid_t location,
    const std::string& name,
    const std::string& context) {
    H5L_info_t info{};
    if (H5Lget_info(location, name.c_str(), &info, H5P_DEFAULT) < 0) {
        throw std::runtime_error(
            "Failed to query HDF5 link type for " + context);
    }
    if (info.type != H5L_TYPE_HARD) {
        throw std::runtime_error(
            context + " must be stored as a file-local HDF5 hard link");
    }
}

void write_attr(
    hid_t location,
    const std::string& name,
    hid_t type,
    const void* data) {
    auto space = H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR),
        "scalar attribute space " + name);
    auto attr = H5AttributeHandle::checked(
        H5Acreate2(
            location,
            name.c_str(),
            detail::canonical_snapshot_file_type(
                type, "snapshot attribute " + name),
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        "attribute " + name);
    check_hdf5(
        H5Awrite(attr.get(), type, data),
        "write attribute " + name);
}

void write_attr_arr(
    hid_t location,
    const std::string& name,
    hid_t type,
    const void* data,
    hsize_t count) {
    const hsize_t dims[1] = {count};
    auto space = H5SpaceHandle::checked(
        H5Screate_simple(1, dims, nullptr),
        "array attribute space " + name);
    auto attr = H5AttributeHandle::checked(
        H5Acreate2(
            location,
            name.c_str(),
            detail::canonical_snapshot_file_type(
                type, "snapshot array attribute " + name),
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        "attribute " + name);
    check_hdf5(
        H5Awrite(attr.get(), type, data),
        "write attribute " + name);
}

void write_string_attr(
    hid_t location,
    const std::string& name,
    const std::string& value) {
    auto space = H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR),
        "string attribute space " + name);
    auto type = H5TypeHandle::checked(
        H5Tcopy(H5T_C_S1),
        "string attribute type " + name);
    check_hdf5(
        H5Tset_cset(type.get(), H5T_CSET_UTF8),
        "set UTF-8 string attribute character set " + name);
    check_hdf5(
        H5Tset_size(type.get(), value.size() + 1),
        "set string attribute size " + name);
    check_hdf5(
        H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
        "set string attribute padding " + name);
    auto attr = H5AttributeHandle::checked(
        H5Acreate2(
            location,
            name.c_str(),
            type.get(),
            space.get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        "string attribute " + name);
    check_hdf5(
        H5Awrite(attr.get(), type.get(), value.c_str()),
        "write string attribute " + name);
}

bool is_six_element_snapshot_attribute(const std::string& name) {
    return name == schema::ATTR_MASS_TABLE
        || name == schema::ATTR_NUM_PART_THIS_FILE
        || name == schema::ATTR_NUM_PART_TOTAL;
}

void require_snapshot_attribute_shape(
    hid_t attribute,
    const std::string& name) {
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute),
        "attribute dataspace " + name);
    if (is_six_element_snapshot_attribute(name)) {
        if (H5Sget_simple_extent_ndims(space.get()) != 1) {
            throw std::runtime_error(
                "Snapshot array attribute must have rank one: " + name);
        }
        hsize_t dimensions[1]{0};
        check_hdf5(
            H5Sget_simple_extent_dims(space.get(), dimensions, nullptr),
            "read snapshot attribute dimensions " + name);
        if (dimensions[0] != 6) {
            throw std::runtime_error(
                "Snapshot array attribute length mismatch: " + name);
        }
        return;
    }
    if (H5Sget_simple_extent_type(space.get()) != H5S_SCALAR) {
        throw std::runtime_error(
            "Snapshot attribute must be scalar: " + name);
    }
}

void read_attr(
    hid_t location,
    const std::string& name,
    hid_t type,
    void* data) {
    auto attr = H5AttributeHandle::checked(
        H5Aopen(location, name.c_str(), H5P_DEFAULT),
        "attribute " + name);
    require_snapshot_attribute_shape(attr.get(), name);
    require_numeric_attribute_representation(
        attr.get(), type, "snapshot attribute " + name);
    check_hdf5(
        H5Aread(attr.get(), type, data),
        "read attribute " + name);
}

std::string read_string_attr(
    hid_t location,
    const std::string& name,
    std::size_t maximum_logical_bytes,
    bool require_nonempty = false) {
    return read_bounded_fixed_string_attribute(
        location,
        name,
        maximum_logical_bytes,
        require_nonempty,
        "Native snapshot");
}

void require_string_attr_value(
    hid_t location,
    const std::string& name,
    const std::string& expected,
    const std::string& context) {
    if (!attribute_exists(location, name)) {
        throw std::runtime_error(
            context + " is missing required attribute " + name);
    }
    const std::string actual = read_string_attr(
        location, name, expected.size());
    if (actual != expected) {
        throw std::runtime_error(
            context + " attribute " + name + " is '" + actual
            + "' but expected '" + expected + "'");
    }
}

std::string require_nonempty_string_attr(
    hid_t location,
    const std::string& name,
    const std::string& context,
    std::size_t maximum_logical_bytes) {
    if (!attribute_exists(location, name)) {
        throw std::runtime_error(
            context + " is missing required attribute " + name);
    }
    return read_string_attr(
        location, name, maximum_logical_bytes, true);
}

H5DatasetHandle create_dataset(
    hid_t group,
    const std::string& name,
    hid_t type,
    hid_t space) {
    return H5DatasetHandle::checked(
        H5Dcreate2(
            group,
            name.c_str(),
            detail::canonical_snapshot_file_type(
                type, "snapshot dataset " + name),
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        "dataset " + name);
}

void require_inline_dataset_storage(
    hid_t dataset,
    const std::string& name) {
    auto creation = H5PropertyHandle::checked(
        H5Dget_create_plist(dataset),
        "dataset creation property list " + name);
    const H5D_layout_t layout = H5Pget_layout(creation.get());
    if (layout == H5D_LAYOUT_ERROR) {
        throw std::runtime_error(
            "Failed to query dataset storage layout: " + name);
    }
    if (layout == H5D_VIRTUAL) {
        throw std::runtime_error(
            "Native snapshot dataset uses virtual storage outside the snapshot file: "
            + name);
    }
    const int external_count = H5Pget_external_count(creation.get());
    if (external_count < 0) {
        throw std::runtime_error(
            "Failed to query external dataset storage: " + name);
    }
    if (external_count != 0) {
        throw std::runtime_error(
            "Native snapshot dataset uses external storage outside the snapshot file: "
            + name);
    }
}

H5DatasetHandle open_dataset(
    hid_t group,
    const std::string& name) {
    require_hard_link(group, name, "Native snapshot dataset " + name);
    auto dataset = H5DatasetHandle::checked(
        H5Dopen2(group, name.c_str(), H5P_DEFAULT),
        "dataset " + name);
    require_inline_dataset_storage(dataset.get(), name);
    return dataset;
}

void require_vector_dataset_shape(
    hid_t dataset,
    const std::string& name,
    std::size_t expected_n) {
    auto space = H5SpaceHandle::checked(
        H5Dget_space(dataset),
        "dataspace " + name);
    if (H5Sget_simple_extent_ndims(space.get()) != 2) {
        throw std::runtime_error(name + " dataset must have rank 2");
    }
    hsize_t dims[2] = {0, 0};
    check_hdf5(
        H5Sget_simple_extent_dims(space.get(), dims, nullptr),
        "read dimensions " + name);
    if (dims[0] != static_cast<hsize_t>(expected_n) || dims[1] != 3) {
        throw std::runtime_error(name + " dataset shape mismatch");
    }
}

void require_scalar_dataset_shape(
    hid_t dataset,
    const std::string& name,
    std::size_t expected_n) {
    auto space = H5SpaceHandle::checked(
        H5Dget_space(dataset),
        "dataspace " + name);
    if (H5Sget_simple_extent_ndims(space.get()) != 1) {
        throw std::runtime_error(name + " dataset must have rank 1");
    }
    hsize_t dims[1] = {0};
    check_hdf5(
        H5Sget_simple_extent_dims(space.get(), dims, nullptr),
        "read dimensions " + name);
    if (dims[0] != static_cast<hsize_t>(expected_n)) {
        throw std::runtime_error(name + " dataset length mismatch");
    }
}

std::size_t checked_batch_particles(
    const config::SimulationParameters& config,
    std::size_t n) {
    const std::uint64_t configured =
        config.get_output().snapshot_batch_particles;
    if (configured == 0) {
        throw std::invalid_argument(
            "Snapshot batch size must be positive");
    }
    if (configured > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "Snapshot batch size does not fit size_t");
    }
    return std::max<std::size_t>(
        1,
        std::min<std::size_t>(
            n == 0 ? 1 : n,
            static_cast<std::size_t>(configured)));
}
