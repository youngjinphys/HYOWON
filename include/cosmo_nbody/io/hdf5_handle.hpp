#pragma once

#include <hdf5.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace io {

namespace detail {

inline std::atomic<std::uint64_t> hdf5_close_failure_epoch{0};

inline void record_hdf5_close_failure() noexcept {
    std::uint64_t observed = hdf5_close_failure_epoch.load(
        std::memory_order_relaxed);
    const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    while (observed != maximum
           && !hdf5_close_failure_epoch.compare_exchange_weak(
               observed,
               observed + 1,
               std::memory_order_release,
               std::memory_order_relaxed)) {
    }
}

inline void require_matching_pad(
    hid_t stored_type,
    hid_t expected_type,
    const std::string& context) {
    H5T_pad_t stored_lsb = H5T_PAD_ERROR;
    H5T_pad_t stored_msb = H5T_PAD_ERROR;
    H5T_pad_t expected_lsb = H5T_PAD_ERROR;
    H5T_pad_t expected_msb = H5T_PAD_ERROR;
    if (H5Tget_pad(stored_type, &stored_lsb, &stored_msb) < 0
        || H5Tget_pad(expected_type, &expected_lsb, &expected_msb) < 0) {
        throw std::runtime_error(
            "Failed to query numeric HDF5 padding: " + context);
    }
    if (stored_lsb != expected_lsb || stored_msb != expected_msb) {
        throw std::runtime_error(
            "Numeric HDF5 padding differs from the native schema: " + context);
    }
}

inline void require_matching_float_layout(
    hid_t stored_type,
    hid_t expected_type,
    const std::string& context) {
    std::size_t stored_sign = 0;
    std::size_t stored_exponent = 0;
    std::size_t stored_exponent_size = 0;
    std::size_t stored_mantissa = 0;
    std::size_t stored_mantissa_size = 0;
    std::size_t expected_sign = 0;
    std::size_t expected_exponent = 0;
    std::size_t expected_exponent_size = 0;
    std::size_t expected_mantissa = 0;
    std::size_t expected_mantissa_size = 0;
    if (H5Tget_fields(
            stored_type,
            &stored_sign,
            &stored_exponent,
            &stored_exponent_size,
            &stored_mantissa,
            &stored_mantissa_size) < 0
        || H5Tget_fields(
               expected_type,
               &expected_sign,
               &expected_exponent,
               &expected_exponent_size,
               &expected_mantissa,
               &expected_mantissa_size) < 0) {
        throw std::runtime_error(
            "Failed to query floating HDF5 field layout: " + context);
    }
    if (stored_sign != expected_sign
        || stored_exponent != expected_exponent
        || stored_exponent_size != expected_exponent_size
        || stored_mantissa != expected_mantissa
        || stored_mantissa_size != expected_mantissa_size
        || H5Tget_ebias(stored_type) != H5Tget_ebias(expected_type)
        || H5Tget_norm(stored_type) != H5Tget_norm(expected_type)
        || H5Tget_inpad(stored_type) != H5Tget_inpad(expected_type)) {
        throw std::runtime_error(
            "Floating HDF5 bit layout differs from the native schema: " + context);
    }
}

// Attribute vs dataset admission differs for integers only:
// - Attributes may be widened in memory (e.g. stored int32 flags read as
//   int64 for range checking). Class and signedness remain semantic gates;
//   width conversion is allowed only when every stored value is representable
//   in the destination type. HDF5 saturation/wrap of out-of-range integers is
//   rejected before narrowing.
// - Datasets bind storage width and bit layout so particle payloads cannot
//   silently promote/truncate under the native schema.
// Floating attributes and datasets both require full precision and IEEE-style
// field layout. Byte order is never compared: endian-only file types remain
// admissible for both paths.
enum class NumericAdmissionKind {
    Attribute,
    Dataset
};

inline void require_matching_precision(
    hid_t stored_type,
    hid_t expected_type,
    const std::string& context) {
    const std::size_t stored_size = H5Tget_size(stored_type);
    const std::size_t expected_size = H5Tget_size(expected_type);
    const std::size_t stored_precision = H5Tget_precision(stored_type);
    const std::size_t expected_precision = H5Tget_precision(expected_type);
    if (stored_size == 0 || expected_size == 0
        || stored_precision == 0 || expected_precision == 0) {
        throw std::runtime_error(
            "Failed to query numeric HDF5 size/precision: " + context);
    }
    if (stored_size != expected_size
        || stored_precision != expected_precision
        || H5Tget_offset(stored_type) != H5Tget_offset(expected_type)) {
        throw std::runtime_error(
            "Numeric HDF5 precision differs from the native schema: " + context);
    }
    require_matching_pad(stored_type, expected_type, context);
}

inline void require_native_numeric_type(
    hid_t stored_type,
    hid_t expected_type,
    const std::string& context,
    NumericAdmissionKind kind) {
    const H5T_class_t stored_class = H5Tget_class(stored_type);
    const H5T_class_t expected_class = H5Tget_class(expected_type);
    if (stored_class == H5T_NO_CLASS || expected_class == H5T_NO_CLASS) {
        throw std::runtime_error(
            "Failed to query numeric HDF5 datatype class: " + context);
    }
    if ((expected_class != H5T_INTEGER && expected_class != H5T_FLOAT)
        || stored_class != expected_class) {
        throw std::runtime_error(
            "Numeric HDF5 type class mismatch: " + context);
    }

    if (expected_class == H5T_INTEGER) {
        const H5T_sign_t stored_sign = H5Tget_sign(stored_type);
        const H5T_sign_t expected_sign = H5Tget_sign(expected_type);
        if (stored_sign == H5T_SGN_ERROR || expected_sign == H5T_SGN_ERROR) {
            throw std::runtime_error(
                "Failed to query integer HDF5 signedness: " + context);
        }
        if (stored_sign != expected_sign) {
            throw std::runtime_error(
                "Integer HDF5 signedness differs from the native schema: " + context);
        }
        if (kind == NumericAdmissionKind::Dataset) {
            require_matching_precision(stored_type, expected_type, context);
        }
        return;
    }

    require_matching_precision(stored_type, expected_type, context);
    require_matching_float_layout(stored_type, expected_type, context);
}

struct IntegerConversionException {
    bool raised{false};
};

inline H5T_conv_ret_t abort_lossy_integer_conversion(
    H5T_conv_except_t,
    hid_t,
    hid_t,
    void*,
    void*,
    void* user_data) noexcept {
    if (user_data != nullptr) {
        static_cast<IntegerConversionException*>(user_data)->raised = true;
    }
    return H5T_CONV_ABORT;
}

inline bool stored_integer_bit(
    const unsigned char* bytes,
    std::size_t size,
    H5T_order_t order,
    std::size_t bit) noexcept {
    const std::size_t byte_from_lsb = bit / 8;
    const std::size_t byte_index = order == H5T_ORDER_BE
        ? size - byte_from_lsb - 1
        : byte_from_lsb;
    return (bytes[byte_index] & (1u << (bit % 8))) != 0;
}

inline void require_stored_integer_values_representable(
    const unsigned char* bytes,
    std::size_t buffer_size,
    std::size_t count,
    hid_t stored_type,
    hid_t destination_type,
    const std::string& context) {
    const std::size_t stored_size = H5Tget_size(stored_type);
    const std::size_t stored_precision = H5Tget_precision(stored_type);
    const std::size_t stored_offset = H5Tget_offset(stored_type);
    const std::size_t destination_precision =
        H5Tget_precision(destination_type);
    const H5T_sign_t stored_sign = H5Tget_sign(stored_type);
    const H5T_sign_t destination_sign = H5Tget_sign(destination_type);
    const H5T_order_t stored_order = H5Tget_order(stored_type);
    H5T_pad_t stored_lsb_pad = H5T_PAD_ERROR;
    H5T_pad_t stored_msb_pad = H5T_PAD_ERROR;
    const bool pad_query_ok =
        H5Tget_pad(stored_type, &stored_lsb_pad, &stored_msb_pad) >= 0;
    if (stored_size == 0
        || stored_size > std::numeric_limits<std::size_t>::max() / 8
        || stored_precision == 0
        || destination_precision == 0
        || stored_sign == H5T_SGN_ERROR
        || destination_sign == H5T_SGN_ERROR
        || stored_sign != destination_sign
        || (stored_order != H5T_ORDER_LE && stored_order != H5T_ORDER_BE
            && !(stored_order == H5T_ORDER_NONE && stored_size == 1))
        || !pad_query_ok
        || stored_lsb_pad == H5T_PAD_ERROR
        || stored_msb_pad == H5T_PAD_ERROR) {
        throw std::runtime_error(
            "Invalid integer HDF5 attribute bit layout: " + context);
    }
    const std::size_t stored_bits = stored_size * 8;
    if (stored_offset > stored_bits
        || stored_precision > stored_bits - stored_offset
        || count > std::numeric_limits<std::size_t>::max() / stored_size
        || buffer_size < count * stored_size) {
        throw std::runtime_error(
            "Invalid integer HDF5 attribute storage extent: " + context);
    }
    if (stored_precision <= destination_precision) {
        return;
    }

    // Bits outside [offset, offset + precision) are padding by the stored
    // HDF5 type and never contribute to the represented value, regardless of
    // whether the declared pad policy is zero, one, or background.
    for (std::size_t index = 0; index < count; ++index) {
        const unsigned char* value = bytes + index * stored_size;
        bool representable = true;
        if (stored_sign == H5T_SGN_NONE) {
            for (std::size_t bit = destination_precision;
                 bit < stored_precision;
                 ++bit) {
                if (stored_integer_bit(
                        value,
                        stored_size,
                        stored_order,
                        stored_offset + bit)) {
                    representable = false;
                    break;
                }
            }
        } else {
            const bool sign_bit = stored_integer_bit(
                value,
                stored_size,
                stored_order,
                stored_offset + stored_precision - 1);
            for (std::size_t bit = destination_precision - 1;
                 bit < stored_precision;
                 ++bit) {
                if (stored_integer_bit(
                        value,
                        stored_size,
                        stored_order,
                        stored_offset + bit)
                    != sign_bit) {
                    representable = false;
                    break;
                }
            }
        }
        if (!representable) {
            throw std::runtime_error(
                "Integer HDF5 attribute cannot be converted losslessly: "
                + context);
        }
    }
}

} // namespace detail

inline std::uint64_t current_hdf5_close_failure_epoch() noexcept {
    return detail::hdf5_close_failure_epoch.load(std::memory_order_acquire);
}

inline void require_no_hdf5_close_failures_since(
    std::uint64_t baseline,
    const std::string& context) {
    if (current_hdf5_close_failure_epoch() != baseline) {
        throw std::runtime_error(
            "HDF5 object close failed before publication: " + context);
    }
}

template <herr_t (*Closer)(hid_t)>
class Hdf5Handle {
public:
    Hdf5Handle() = default;
    explicit Hdf5Handle(hid_t id) noexcept : id_(id) {}

    static Hdf5Handle checked(hid_t id, const std::string& context) {
        if (id < 0) {
            throw std::runtime_error(
                "HDF5 handle creation/open failed: " + context);
        }
        return Hdf5Handle(id);
    }

    ~Hdf5Handle() { reset(); }

    Hdf5Handle(const Hdf5Handle&) = delete;
    Hdf5Handle& operator=(const Hdf5Handle&) = delete;

    Hdf5Handle(Hdf5Handle&& other) noexcept
        : id_(std::exchange(other.id_, -1)) {}

    Hdf5Handle& operator=(Hdf5Handle&& other) noexcept {
        if (this != &other) {
            reset();
            id_ = std::exchange(other.id_, -1);
        }
        return *this;
    }

    hid_t get() const noexcept { return id_; }
    explicit operator bool() const noexcept { return id_ >= 0; }

    void reset(hid_t replacement = -1) noexcept {
        if (id_ >= 0 && Closer(id_) < 0) {
            detail::record_hdf5_close_failure();
        }
        id_ = replacement;
    }

private:
    hid_t id_{-1};
};

using H5FileHandle = Hdf5Handle<H5Fclose>;
using H5GroupHandle = Hdf5Handle<H5Gclose>;
using H5DatasetHandle = Hdf5Handle<H5Dclose>;
using H5SpaceHandle = Hdf5Handle<H5Sclose>;
using H5AttributeHandle = Hdf5Handle<H5Aclose>;
using H5TypeHandle = Hdf5Handle<H5Tclose>;
using H5PropertyHandle = Hdf5Handle<H5Pclose>;

inline void check_hdf5(herr_t status, const std::string& context) {
    if (status < 0) {
        throw std::runtime_error(
            "HDF5 operation failed: " + context);
    }
}

// Default-format HDF5 groups keep attributes in compact object-header
// messages. A fixed-length provenance string near the public persisted-text
// bound can exceed that representation even though H5Tset_size accepted it.
// Tracking attribute creation order enables the modern object representation;
// a zero phase-change threshold forces dense storage from the first attribute
// without changing the group path, attribute name, or datatype.
inline H5PropertyHandle provenance_group_creation_properties(
    const std::string& context) {
    auto properties = H5PropertyHandle::checked(
        H5Pcreate(H5P_GROUP_CREATE),
        context + " property list");
    check_hdf5(
        H5Pset_attr_creation_order(
            properties.get(),
            H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED),
        context + " attribute creation-order tracking");
    check_hdf5(
        H5Pset_attr_phase_change(properties.get(), 0U, 0U),
        context + " dense attribute storage");
    return properties;
}

inline void require_numeric_attribute_representation(
    hid_t attribute,
    hid_t memory_type,
    const std::string& context) {
    auto stored_type = H5TypeHandle::checked(
        H5Aget_type(attribute), context + " stored datatype");
    detail::require_native_numeric_type(
        stored_type.get(),
        memory_type,
        context,
        detail::NumericAdmissionKind::Attribute);
}

inline herr_t read_integer_attribute_lossless(
    hid_t attribute,
    hid_t memory_type,
    void* data,
    const std::string& context) {
    const std::size_t dest_size = H5Tget_size(memory_type);
    const H5T_sign_t dest_sign = H5Tget_sign(memory_type);
    if (dest_size == 0 || dest_sign == H5T_SGN_ERROR) {
        throw std::runtime_error(
            "Failed to query integer destination type: " + context);
    }
    if (dest_size != 1 && dest_size != 2 && dest_size != 4 && dest_size != 8) {
        throw std::runtime_error(
            "Unsupported integer destination width: " + context);
    }

    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute), context + " dataspace");
    const hssize_t npoints = H5Sget_simple_extent_npoints(space.get());
    if (npoints < 0) {
        throw std::runtime_error(
            "Failed to query integer attribute length: " + context);
    }
    if (static_cast<std::uintmax_t>(npoints)
        > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "Integer attribute length exceeds addressable memory: " + context);
    }
    const std::size_t count = static_cast<std::size_t>(npoints);
    if (count == 0) {
        return 0;
    }
    if (data == nullptr) {
        throw std::runtime_error(
            "Integer attribute destination is null: " + context);
    }

    auto stored_type = H5TypeHandle::checked(
        H5Aget_type(attribute), context + " stored datatype");
    const std::size_t stored_size = H5Tget_size(stored_type.get());
    if (stored_size == 0) {
        throw std::runtime_error(
            "Failed to query integer stored type size: " + context);
    }
    const std::size_t conversion_size = std::max(stored_size, dest_size);
    if (count > std::numeric_limits<std::size_t>::max() / conversion_size) {
        throw std::runtime_error(
            "Integer attribute conversion buffer size overflow: " + context);
    }
    std::vector<unsigned char> conversion_buffer(
        count * conversion_size, 0);
    const herr_t read_status = ::H5Aread(
        attribute, stored_type.get(), conversion_buffer.data());
    if (read_status < 0) {
        return read_status;
    }
    detail::require_stored_integer_values_representable(
        conversion_buffer.data(),
        conversion_buffer.size(),
        count,
        stored_type.get(),
        memory_type,
        context);
    auto transfer = H5PropertyHandle::checked(
        H5Pcreate(H5P_DATASET_XFER), context + " conversion properties");
    detail::IntegerConversionException conversion_exception;
    check_hdf5(
        H5Pset_type_conv_cb(
            transfer.get(),
            detail::abort_lossy_integer_conversion,
            &conversion_exception),
        "install lossless integer conversion guard: " + context);
    herr_t conversion_status = -1;
    H5E_BEGIN_TRY {
        conversion_status = H5Tconvert(
            stored_type.get(),
            memory_type,
            count,
            conversion_buffer.data(),
            nullptr,
            transfer.get());
    } H5E_END_TRY;
    if (conversion_status < 0) {
        const std::string reason = conversion_exception.raised
            ? "Integer HDF5 attribute cannot be converted losslessly: "
            : "Integer HDF5 attribute conversion failed: ";
        throw std::runtime_error(reason + context);
    }

    if (count > std::numeric_limits<std::size_t>::max() / dest_size) {
        throw std::runtime_error(
            "Integer attribute destination size overflow: " + context);
    }
    std::memcpy(data, conversion_buffer.data(), count * dest_size);
    return 0;
}

inline void require_numeric_dataset_representation(
    hid_t dataset,
    hid_t memory_type,
    const std::string& context) {
    auto stored_type = H5TypeHandle::checked(
        H5Dget_type(dataset), context + " stored datatype");
    detail::require_native_numeric_type(
        stored_type.get(),
        memory_type,
        context,
        detail::NumericAdmissionKind::Dataset);
}

// Project-local overloads: unqualified reads inside cosmo_nbody::io pass through
// native-schema admission before HDF5 conversion. Byte order may differ for both
// attributes and datasets. Integer attributes may also differ in storage width;
// datasets and floating attributes bind precision, padding, and float layout.
// Explicit ::H5Aread/::H5Dread is reserved for a deliberately conversion-permissive
// external format and must not be used for native simulation state.
inline herr_t H5Aread(
    hid_t attribute,
    hid_t memory_type,
    void* data,
    const std::string& context) {
    const H5T_class_t expected_class = H5Tget_class(memory_type);
    if (expected_class == H5T_NO_CLASS) {
        throw std::runtime_error(
            "Failed to query HDF5 attribute memory datatype class: " + context);
    }
    if (expected_class == H5T_INTEGER || expected_class == H5T_FLOAT) {
        require_numeric_attribute_representation(attribute, memory_type, context);
    }
    if (expected_class == H5T_INTEGER) {
        return read_integer_attribute_lossless(
            attribute, memory_type, data, context);
    }
    return ::H5Aread(attribute, memory_type, data);
}

inline herr_t H5Aread(hid_t attribute, hid_t memory_type, void* data) {
    return H5Aread(
        attribute, memory_type, data, "native-schema attribute read");
}

inline herr_t H5Dread(
    hid_t dataset,
    hid_t memory_type,
    hid_t memory_space,
    hid_t file_space,
    hid_t transfer_properties,
    void* data,
    const std::string& context) {
    const H5T_class_t expected_class = H5Tget_class(memory_type);
    if (expected_class == H5T_NO_CLASS) {
        throw std::runtime_error(
            "Failed to query HDF5 dataset memory datatype class: " + context);
    }
    if (expected_class == H5T_INTEGER || expected_class == H5T_FLOAT) {
        require_numeric_dataset_representation(dataset, memory_type, context);
    }
    return ::H5Dread(
        dataset,
        memory_type,
        memory_space,
        file_space,
        transfer_properties,
        data);
}

inline herr_t H5Dread(
    hid_t dataset,
    hid_t memory_type,
    hid_t memory_space,
    hid_t file_space,
    hid_t transfer_properties,
    void* data) {
    return H5Dread(
        dataset,
        memory_type,
        memory_space,
        file_space,
        transfer_properties,
        data,
        "native-schema dataset read");
}

} // namespace io
} // namespace cosmo_nbody
