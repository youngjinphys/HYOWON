#pragma once

#include "cosmo_nbody/io/hdf5_handle.hpp"

#include <hdf5.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody::io {

// Read one scalar fixed-length HDF5 string only after its advertised storage
// size has passed the caller's schema bound. maximum_logical_bytes excludes an
// optional trailing NUL byte.
inline std::string read_bounded_fixed_string_attribute(
    hid_t location,
    const std::string& name,
    std::size_t maximum_logical_bytes,
    bool require_nonempty,
    std::string_view context) {
    const htri_t exists = H5Aexists(location, name.c_str());
    if (exists < 0) {
        throw std::runtime_error(
            "Failed to query " + std::string(context) + " attribute: "
            + name);
    }
    if (exists == 0) {
        throw std::runtime_error(
            std::string(context) + " attribute is missing: " + name);
    }
    auto attribute = H5AttributeHandle::checked(
        H5Aopen(location, name.c_str(), H5P_DEFAULT),
        std::string(context) + " attribute " + name);
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute.get()),
        std::string(context) + " attribute dataspace " + name);
    if (H5Sget_simple_extent_type(space.get()) != H5S_SCALAR) {
        throw std::runtime_error(
            std::string(context) + " string attribute must be scalar: "
            + name);
    }
    auto type = H5TypeHandle::checked(
        H5Aget_type(attribute.get()),
        std::string(context) + " attribute type " + name);
    if (H5Tget_class(type.get()) != H5T_STRING) {
        throw std::runtime_error(
            std::string(context) + " attribute must use string storage: "
            + name);
    }
    const htri_t variable = H5Tis_variable_str(type.get());
    if (variable < 0) {
        throw std::runtime_error(
            "Failed to query HDF5 string representation: " + name);
    }
    if (variable > 0) {
        throw std::runtime_error(
            std::string(context)
            + " string attribute must use fixed-length storage: " + name);
    }
    const std::size_t stored_bytes = H5Tget_size(type.get());
    if (stored_bytes == 0U) {
        throw std::runtime_error(
            std::string(context) + " string attribute has zero size: "
            + name);
    }
    const std::size_t maximum_stored_bytes =
        maximum_logical_bytes == std::numeric_limits<std::size_t>::max()
        ? maximum_logical_bytes
        : maximum_logical_bytes + 1U;
    if (stored_bytes > maximum_stored_bytes) {
        throw std::runtime_error(
            std::string(context)
            + " string attribute exceeds its allocation-before-admission bound: "
            + name);
    }

    std::string value(stored_bytes, '\0');
    check_hdf5(
        H5Aread(attribute.get(), type.get(), value.data()),
        std::string(context) + " read string attribute " + name);
    while (!value.empty() && value.back() == '\0') value.pop_back();
    if (value.size() > maximum_logical_bytes) {
        throw std::runtime_error(
            std::string(context) + " string attribute exceeds its logical bound: "
            + name);
    }
    if (require_nonempty && value.empty()) {
        throw std::runtime_error(
            std::string(context) + " string attribute must not be empty: "
            + name);
    }
    return value;
}

} // namespace cosmo_nbody::io
