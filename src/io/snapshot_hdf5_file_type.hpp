#pragma once

#include <hdf5.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::io::detail {

// Persist snapshot numerics with explicit little-endian file types. HDF5
// converts from the native in-memory type during writes.
inline hid_t canonical_snapshot_file_type(
    hid_t memory_type,
    const std::string& context) {
    const H5T_class_t type_class = H5Tget_class(memory_type);
    if (type_class == H5T_NO_CLASS) {
        throw std::runtime_error(
            "Failed to inspect HDF5 memory datatype for " + context);
    }
    const std::size_t bytes = H5Tget_size(memory_type);
    if (bytes == 0) {
        throw std::runtime_error(
            "HDF5 memory datatype has zero size for " + context);
    }

    if (type_class == H5T_FLOAT) {
        if (bytes == 4) return H5T_IEEE_F32LE;
        if (bytes == 8) return H5T_IEEE_F64LE;
    } else if (type_class == H5T_INTEGER) {
        const H5T_sign_t sign = H5Tget_sign(memory_type);
        if (sign == H5T_SGN_ERROR) {
            throw std::runtime_error(
                "Failed to inspect HDF5 integer sign for " + context);
        }
        if (sign == H5T_SGN_NONE) {
            if (bytes == 1) return H5T_STD_U8LE;
            if (bytes == 2) return H5T_STD_U16LE;
            if (bytes == 4) return H5T_STD_U32LE;
            if (bytes == 8) return H5T_STD_U64LE;
        } else if (sign == H5T_SGN_2) {
            if (bytes == 1) return H5T_STD_I8LE;
            if (bytes == 2) return H5T_STD_I16LE;
            if (bytes == 4) return H5T_STD_I32LE;
            if (bytes == 8) return H5T_STD_I64LE;
        }
    }

    throw std::runtime_error(
        "Snapshot numeric datatype has no canonical little-endian mapping for "
        + context);
}

} // namespace cosmo_nbody::io::detail
