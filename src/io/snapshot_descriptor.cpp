#include "cosmo_nbody/io/snapshot_descriptor.hpp"

#include "cosmo_nbody/cosmology/flat_matter_lambda.hpp"
#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/io/bounded_hdf5_string.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/hdf5_handle.hpp"
#include "cosmo_nbody/io/metadata.hpp"
#include "cosmo_nbody/io/output_schema.hpp"
#include "cosmo_nbody/io/snapshot_physics.hpp"
#include "cosmo_nbody/io/verified_snapshot_source.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__APPLE__)
#include <xlocale.h>
#else
#include <locale.h>
#endif

namespace cosmo_nbody::io {
namespace {

bool nearly_equal(double lhs, double rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) return false;
    if (lhs == rhs) return true;
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    if (scale == 0.0) return true;
    return std::abs(lhs - rhs)
        <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

void require_hard_link(hid_t location, const std::string& name) {
    H5L_info_t info{};
    if (H5Lget_info(location, name.c_str(), &info, H5P_DEFAULT) < 0) {
        throw std::runtime_error("Failed to query native snapshot link: " + name);
    }
    if (info.type != H5L_TYPE_HARD) {
        throw std::runtime_error(
            "Native snapshot link must be a file-local hard link: " + name);
    }
}

bool attribute_exists(hid_t location, const std::string& name) {
    const htri_t result = H5Aexists(location, name.c_str());
    if (result < 0) {
        throw std::runtime_error("Failed to query snapshot attribute: " + name);
    }
    return result > 0;
}

void require_scalar_space(hid_t attribute, const std::string& name) {
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute), "attribute dataspace " + name);
    if (H5Sget_simple_extent_type(space.get()) != H5S_SCALAR) {
        throw std::runtime_error("Snapshot attribute must be scalar: " + name);
    }
}

void require_numeric_type(
    hid_t attribute,
    const std::string& name,
    hid_t native_type) {
    require_numeric_attribute_representation(
        attribute,
        native_type,
        "snapshot descriptor attribute " + name);
}

template <typename T>
T read_scalar(hid_t location, const std::string& name, hid_t native_type) {
    if (!attribute_exists(location, name)) {
        throw std::runtime_error("Snapshot is missing required attribute: " + name);
    }
    auto attribute = H5AttributeHandle::checked(
        H5Aopen(location, name.c_str(), H5P_DEFAULT), "attribute " + name);
    require_scalar_space(attribute.get(), name);
    require_numeric_type(attribute.get(), name, native_type);
    T value{};
    check_hdf5(
        H5Aread(
            attribute.get(),
            native_type,
            &value,
            "snapshot descriptor attribute " + name),
        "read snapshot attribute " + name);
    return value;
}

template <typename T, std::size_t N>
std::array<T, N> read_array(
    hid_t location,
    const std::string& name,
    hid_t native_type) {
    if (!attribute_exists(location, name)) {
        throw std::runtime_error("Snapshot is missing required attribute: " + name);
    }
    auto attribute = H5AttributeHandle::checked(
        H5Aopen(location, name.c_str(), H5P_DEFAULT), "attribute " + name);
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute.get()), "attribute dataspace " + name);
    if (H5Sget_simple_extent_ndims(space.get()) != 1) {
        throw std::runtime_error("Snapshot array attribute must have rank one: " + name);
    }
    hsize_t dimensions[1]{0};
    check_hdf5(
        H5Sget_simple_extent_dims(space.get(), dimensions, nullptr),
        "read snapshot attribute dimensions " + name);
    if (dimensions[0] != static_cast<hsize_t>(N)) {
        throw std::runtime_error("Snapshot array attribute length mismatch: " + name);
    }
    require_numeric_type(attribute.get(), name, native_type);
    std::array<T, N> values{};
    check_hdf5(
        H5Aread(
            attribute.get(),
            native_type,
            values.data(),
            "snapshot descriptor attribute " + name),
        "read snapshot array attribute " + name);
    return values;
}

std::string read_nonempty_string(
    hid_t location,
    const std::string& name,
    std::size_t maximum_logical_bytes) {
    if (!attribute_exists(location, name)) {
        throw std::runtime_error("Snapshot is missing required attribute: " + name);
    }
    return read_bounded_fixed_string_attribute(
        location,
        name,
        maximum_logical_bytes,
        true,
        "Snapshot descriptor");
}

std::string read_required_string(
    hid_t location,
    const std::string& name,
    std::size_t maximum_logical_bytes) {
    if (!attribute_exists(location, name)) {
        throw std::runtime_error("Snapshot is missing required attribute: " + name);
    }
    return read_bounded_fixed_string_attribute(
        location,
        name,
        maximum_logical_bytes,
        false,
        "Snapshot descriptor");
}

bool is_json_number_token(std::string_view token) noexcept;

bool is_hexadecimal_digit(unsigned char character) noexcept {
    return (character >= '0' && character <= '9')
        || (character >= 'a' && character <= 'f')
        || (character >= 'A' && character <= 'F');
}

void scan_canonical_json_string(
    std::string_view json,
    std::size_t& cursor,
    std::size_t limit,
    const char* context) {
    if (cursor >= limit || json[cursor] != '"') {
        throw std::runtime_error(
            std::string(context) + " must begin with a JSON string");
    }
    ++cursor;
    while (cursor < limit) {
        const unsigned char character =
            static_cast<unsigned char>(json[cursor++]);
        if (character == '\\') {
            if (cursor >= limit) {
                throw std::runtime_error(
                    std::string(context)
                    + " has an incomplete JSON escape");
            }
            const unsigned char escaped =
                static_cast<unsigned char>(json[cursor++]);
            if (escaped == 'u') {
                if (limit - cursor < 4U) {
                    throw std::runtime_error(
                        std::string(context)
                        + " has an incomplete Unicode escape");
                }
                for (unsigned int digit = 0; digit < 4U; ++digit) {
                    if (!is_hexadecimal_digit(
                            static_cast<unsigned char>(json[cursor++]))) {
                        throw std::runtime_error(
                            std::string(context)
                            + " has an invalid Unicode escape");
                    }
                }
            } else if (escaped != '"' && escaped != '\\'
                       && escaped != '/' && escaped != 'b'
                       && escaped != 'f' && escaped != 'n'
                       && escaped != 'r' && escaped != 't') {
                throw std::runtime_error(
                    std::string(context) + " has an invalid JSON escape");
            }
        } else if (character == '"') {
            return;
        } else if (character < 0x20U) {
            throw std::runtime_error(
                std::string(context)
                + " contains an unescaped control character");
        }
    }
    throw std::runtime_error(
        std::string(context) + " has an unterminated JSON string");
}

bool is_canonical_json_scalar(std::string_view token) noexcept {
    return token == "true" || token == "false" || token == "null"
        || is_json_number_token(token);
}

void scan_canonical_scalar_until(
    std::string_view json,
    std::size_t& cursor,
    std::size_t limit,
    char delimiter_one,
    char delimiter_two,
    const char* context) {
    const std::size_t start = cursor;
    while (cursor < limit
           && json[cursor] != delimiter_one
           && json[cursor] != delimiter_two) {
        const char character = json[cursor];
        if (character == '{' || character == '}'
            || character == '[' || character == ']'
            || character == '"'
            || character == ' ' || character == '\t'
            || character == '\r' || character == '\n') {
            throw std::runtime_error(
                std::string(context)
                + " contains a non-canonical scalar token");
        }
        ++cursor;
    }
    if (cursor == start) {
        throw std::runtime_error(
            std::string(context) + " contains an empty scalar token");
    }
    if (!is_canonical_json_scalar(json.substr(start, cursor - start))) {
        throw std::runtime_error(
            std::string(context) + " contains an invalid JSON scalar token");
    }
}

void scan_canonical_flat_object(
    std::string_view json,
    std::size_t& cursor,
    std::size_t limit) {
    if (cursor >= limit || json[cursor] != '{') {
        throw std::runtime_error(
            "Snapshot RunMetadataJson FFTW record must be an object");
    }
    ++cursor;
    if (cursor < limit && json[cursor] == '}') {
        throw std::runtime_error(
            "Snapshot RunMetadataJson FFTW record must not be empty");
    }
    std::vector<std::string_view> member_names;
    member_names.reserve(24U);
    while (cursor < limit) {
        const std::size_t key_token_start = cursor;
        scan_canonical_json_string(
            json, cursor, limit, "Snapshot RunMetadataJson FFTW record key");
        const std::string_view member_name = json.substr(
            key_token_start + 1U,
            cursor - key_token_start - 2U);
        if (member_name.find('\\') != std::string_view::npos) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson FFTW record keys must not be escaped");
        }
        if (member_names.size() >= 64U) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson FFTW record has too many members");
        }
        if (std::find(
                member_names.begin(), member_names.end(), member_name)
            != member_names.end()) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson FFTW record has a duplicate key");
        }
        member_names.push_back(member_name);
        if (cursor >= limit || json[cursor] != ':') {
            throw std::runtime_error(
                "Snapshot RunMetadataJson FFTW record key is missing its value delimiter");
        }
        ++cursor;
        if (cursor >= limit) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson FFTW record has an empty value");
        }
        if (json[cursor] == '"') {
            scan_canonical_json_string(
                json,
                cursor,
                limit,
                "Snapshot RunMetadataJson FFTW record value");
        } else {
            scan_canonical_scalar_until(
                json,
                cursor,
                limit,
                ',',
                '}',
                "Snapshot RunMetadataJson FFTW record");
        }
        if (cursor >= limit) break;
        if (json[cursor] == '}') {
            ++cursor;
            return;
        }
        if (json[cursor] != ',') {
            throw std::runtime_error(
                "Snapshot RunMetadataJson FFTW record has an invalid delimiter");
        }
        ++cursor;
        if (cursor >= limit || json[cursor] == '}') {
            throw std::runtime_error(
                "Snapshot RunMetadataJson FFTW record has a trailing comma");
        }
    }
    throw std::runtime_error(
        "Snapshot RunMetadataJson has an unterminated FFTW record object");
}

void scan_canonical_array(
    std::string_view json,
    std::size_t& cursor,
    std::size_t limit,
    bool require_flat_objects) {
    if (cursor >= limit || json[cursor] != '[') {
        throw std::runtime_error(
            "Snapshot RunMetadataJson array scanner lost its opening delimiter");
    }
    ++cursor;
    if (cursor < limit && json[cursor] == ']') {
        ++cursor;
        return;
    }
    while (cursor < limit) {
        if (require_flat_objects) {
            if (json[cursor] != '{') {
                throw std::runtime_error(
                    "Snapshot RunMetadataJson FFTW planning array requires object elements");
            }
            scan_canonical_flat_object(json, cursor, limit);
        } else if (json[cursor] == '"') {
            scan_canonical_json_string(
                json, cursor, limit, "Snapshot RunMetadataJson array value");
        } else {
            scan_canonical_scalar_until(
                json,
                cursor,
                limit,
                ',',
                ']',
                "Snapshot RunMetadataJson array");
        }
        if (cursor >= limit) break;
        if (json[cursor] == ']') {
            ++cursor;
            return;
        }
        if (json[cursor] != ',') {
            throw std::runtime_error(
                "Snapshot RunMetadataJson array has an invalid delimiter");
        }
        ++cursor;
        if (cursor >= limit || json[cursor] == ']') {
            throw std::runtime_error(
                "Snapshot RunMetadataJson array has a trailing comma");
        }
    }
    throw std::runtime_error(
        "Snapshot RunMetadataJson has an unterminated array value");
}

std::string_view canonical_run_metadata_value(
    std::string_view json,
    std::string_view key) {
    if (json.size() < 2 || json.front() != '{' || json.back() != '}') {
        throw std::runtime_error(
            "Snapshot RunMetadataJson is not a compact top-level object");
    }

    std::string_view recovered;
    bool found = false;
    std::vector<std::string_view> top_level_names;
    top_level_names.reserve(96U);
    const std::size_t object_end = json.size() - 1;
    std::size_t cursor = 1;
    while (cursor < object_end) {
        if (json[cursor] != '"') {
            throw std::runtime_error(
                "Snapshot RunMetadataJson has a malformed top-level key");
        }
        const std::size_t name_start = ++cursor;
        while (cursor < object_end && json[cursor] != '"') {
            if (json[cursor] == '\\') {
                throw std::runtime_error(
                    "Snapshot RunMetadataJson top-level keys must not be escaped");
            }
            if (static_cast<unsigned char>(json[cursor]) < 0x20U) {
                throw std::runtime_error(
                    "Snapshot RunMetadataJson top-level key contains a control character");
            }
            ++cursor;
        }
        if (cursor >= object_end) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson has an unterminated top-level key");
        }
        const std::string_view member_name =
            json.substr(name_start, cursor - name_start);
        if (top_level_names.size() >= 256U) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson has too many top-level fields");
        }
        if (std::find(
                top_level_names.begin(), top_level_names.end(), member_name)
            != top_level_names.end()) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson has a duplicate top-level field: "
                + std::string(member_name));
        }
        top_level_names.push_back(member_name);
        ++cursor;
        if (cursor >= object_end || json[cursor] != ':') {
            throw std::runtime_error(
                "Snapshot RunMetadataJson top-level key is missing its value delimiter");
        }
        ++cursor;
        if (cursor >= object_end) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson has an empty top-level value");
        }

        const std::size_t value_start = cursor;
        if (json[cursor] == '"') {
            scan_canonical_json_string(
                json,
                cursor,
                object_end,
                "Snapshot RunMetadataJson top-level value");
        } else if (json[cursor] == '[') {
            scan_canonical_array(
                json,
                cursor,
                object_end,
                member_name == "fftw_planning_records");
        } else {
            scan_canonical_scalar_until(
                json,
                cursor,
                object_end,
                ',',
                '}',
                "Snapshot RunMetadataJson top-level value");
        }

        if (cursor == value_start) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson has an empty top-level value");
        }
        if (member_name == key) {
            if (found) {
                throw std::runtime_error(
                    "Snapshot RunMetadataJson has duplicate field: "
                    + std::string(key));
            }
            recovered = json.substr(value_start, cursor - value_start);
            found = true;
        }
        if (cursor == object_end) break;
        if (json[cursor] != ',') {
            throw std::runtime_error(
                "Snapshot RunMetadataJson top-level value has an invalid delimiter");
        }
        ++cursor;
        if (cursor >= object_end) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson has a trailing comma");
        }
    }
    if (!found) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson is missing required provenance field: "
            + std::string(key));
    }
    return recovered;
}

bool is_json_number_token(std::string_view token) noexcept {
    if (token.empty()) return false;
    std::size_t cursor = 0;
    if (token[cursor] == '-') {
        ++cursor;
        if (cursor == token.size()) return false;
    }
    if (token[cursor] == '0') {
        ++cursor;
        if (cursor < token.size()
            && token[cursor] >= '0' && token[cursor] <= '9') {
            return false;
        }
    } else {
        if (token[cursor] < '1' || token[cursor] > '9') return false;
        do {
            ++cursor;
        } while (cursor < token.size()
                 && token[cursor] >= '0' && token[cursor] <= '9');
    }
    if (cursor < token.size() && token[cursor] == '.') {
        ++cursor;
        const std::size_t fraction_start = cursor;
        while (cursor < token.size()
               && token[cursor] >= '0' && token[cursor] <= '9') {
            ++cursor;
        }
        if (cursor == fraction_start) return false;
    }
    if (cursor < token.size()
        && (token[cursor] == 'e' || token[cursor] == 'E')) {
        ++cursor;
        if (cursor < token.size()
            && (token[cursor] == '+' || token[cursor] == '-')) {
            ++cursor;
        }
        const std::size_t exponent_start = cursor;
        while (cursor < token.size()
               && token[cursor] >= '0' && token[cursor] <= '9') {
            ++cursor;
        }
        if (cursor == exponent_start) return false;
    }
    return cursor == token.size();
}

core::Real canonical_run_metadata_real(
    std::string_view json,
    std::string_view key) {
    const std::string_view token = canonical_run_metadata_value(json, key);
    if (!is_json_number_token(token)) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson provenance field is not numeric: "
            + std::string(key));
    }
    // Apple Clang 14 / libc++ does not provide floating std::from_chars.
    // Parse with a C numeric locale so JSON decimals cannot follow LC_NUMERIC.
    std::string buffer(token);
    locale_t c_locale =
        newlocale(LC_NUMERIC_MASK, "C", static_cast<locale_t>(0));
    if (c_locale == static_cast<locale_t>(0)) {
        throw std::runtime_error(
            "Failed to create C locale while parsing snapshot provenance field: "
            + std::string(key));
    }
    char* parsed_end = nullptr;
    errno = 0;
    const core::Real value = strtod_l(buffer.c_str(), &parsed_end, c_locale);
    freelocale(c_locale);
    if (errno != 0
        || parsed_end != buffer.c_str() + buffer.size()
        || !std::isfinite(value)) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson provenance field is not a finite real: "
            + std::string(key));
    }
    return value;
}

int canonical_run_metadata_int(
    std::string_view json,
    std::string_view key) {
    const std::string_view token = canonical_run_metadata_value(json, key);
    if (!is_json_number_token(token)) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson provenance field is not an integer: "
            + std::string(key));
    }
    int value = 0;
    const char* const begin = token.data();
    const char* const end = token.data() + token.size();
    const auto parsed = std::from_chars(begin, end, value, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson provenance field is not an integer: "
            + std::string(key));
    }
    return value;
}

std::uint64_t canonical_run_metadata_uint64(
    std::string_view json,
    std::string_view key) {
    const std::string_view token = canonical_run_metadata_value(json, key);
    if (!is_json_number_token(token)) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson provenance field is not an unsigned integer: "
            + std::string(key));
    }
    std::uint64_t value = 0;
    const char* const begin = token.data();
    const char* const end = token.data() + token.size();
    const auto parsed = std::from_chars(begin, end, value, 10);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson provenance field is not an unsigned integer: "
            + std::string(key));
    }
    return value;
}

bool canonical_run_metadata_bool(
    std::string_view json,
    std::string_view key) {
    const std::string_view token = canonical_run_metadata_value(json, key);
    if (token == "true") return true;
    if (token == "false") return false;
    throw std::runtime_error(
        "Snapshot RunMetadataJson provenance field is not a boolean: "
        + std::string(key));
}

std::optional<std::uint64_t> nullable_run_metadata_uint64(
    std::string_view json, std::string_view key) {
    // A null value is explicit provenance; an absent field is not current format.
    const auto token = canonical_run_metadata_value(json, key);
    if (token == "null") return std::nullopt;
    return canonical_run_metadata_uint64(json, key);
}

std::string canonical_run_metadata_string(
    std::string_view json,
    std::string_view key) {
    const std::string_view token = canonical_run_metadata_value(json, key);
    if (token.size() < 2 || token.front() != '"' || token.back() != '"') {
        throw std::runtime_error(
            "Snapshot RunMetadataJson provenance field is not a string: "
            + std::string(key));
    }
    if (token.find('\\') != std::string_view::npos) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson recovered provenance string uses unsupported escaping: "
            + std::string(key));
    }
    return std::string(token.substr(1, token.size() - 2));
}

bool is_persisted_power_spectrum_fidelity(const std::string& value) {
    // Empty remains descriptive for a readable snapshot with unknown origin;
    // it is not admitted as an initial condition for further evolution.
    return value.empty() || value == "precision_boltzmann";
}

void require_finite(core::Real value, const char* label) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("Snapshot descriptor field is not finite: ") + label);
    }
}

void require_positive(core::Real value, const char* label) {
    require_finite(value, label);
    if (value <= 0.0) {
        throw std::runtime_error(std::string("Snapshot descriptor field is not positive: ") + label);
    }
}

void require_nonnegative(core::Real value, const char* label) {
    require_finite(value, label);
    if (value < 0.0) {
        throw std::runtime_error(std::string("Snapshot descriptor field is negative: ") + label);
    }
}

[[noreturn]] void comparison_mismatch(
    const char* product,
    const char* field) {
    throw std::invalid_argument(
        std::string(product) + " snapshot match failed: " + field);
}

void require_equal(
    core::Real lhs,
    core::Real rhs,
    const char* product,
    const char* field) {
    if (!nearly_equal(lhs, rhs)) comparison_mismatch(product, field);
}

void require_equal(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* product,
    const char* field) {
    if (lhs != rhs) comparison_mismatch(product, field);
}

void require_equal(
    bool lhs,
    bool rhs,
    const char* product,
    const char* field) {
    if (lhs != rhs) comparison_mismatch(product, field);
}

void require_equal(
    const std::string& lhs,
    const std::string& rhs,
    const char* product,
    const char* field) {
    if (lhs != rhs) comparison_mismatch(product, field);
}

void require_semantic_bindings(
    const SnapshotDescriptor& lhs,
    const SnapshotDescriptor& rhs,
    const char* product) {
    if (lhs.semantic_binding_values != rhs.semantic_binding_values) {
        comparison_mismatch(
            product,
            "phase_space_or_unit_semantic_bindings");
    }
}

std::optional<std::uint64_t> fingerprint_ic_support(std::string_view fingerprint) {
    constexpr std::string_view prefix = "ic_max_mode_per_axis=";
    std::optional<std::uint64_t> result;
    while (!fingerprint.empty()) {
        const auto end = fingerprint.find('\n');
        const auto line = fingerprint.substr(0, end);
        if (line.starts_with(prefix)) {
            const auto token = line.substr(prefix.size());
            std::uint64_t value = 0;
            const auto parsed = std::from_chars(
                token.data(), token.data() + token.size(), value, 10);
            if (result || parsed.ec != std::errc{}
                || parsed.ptr != token.data() + token.size()
                || value == 0 || std::to_string(value) != token) {
                throw std::runtime_error("Snapshot IC support fingerprint is not one canonical positive integer");
            }
            result = value;
        }
        if (end == std::string_view::npos) break;
        fingerprint.remove_prefix(end + 1);
    }
    return result;
}

} // namespace

SnapshotGenerationProvenance read_snapshot_generation_provenance(
    std::string_view run_metadata_json) {
    if (canonical_run_metadata_string(
            run_metadata_json, "product_kind") != "run_metadata"
        || canonical_run_metadata_string(
            run_metadata_json, "lineage") != "direct_simulation") {
        throw std::runtime_error(
            "Snapshot generator provenance requires direct simulation RunMetadataJson");
    }

    const std::string ic_mode = canonical_run_metadata_string(
        run_metadata_json, "ic_mode");
    const std::string provenance_status = canonical_run_metadata_string(
        run_metadata_json, "ic_provenance_status");
    const bool generation_parameters_applied = canonical_run_metadata_bool(
        run_metadata_json, "ic_generation_parameters_applied");

    SnapshotGenerationProvenance provenance;
    provenance.seed = canonical_run_metadata_uint64(
        run_metadata_json, "seed");
    provenance.ic_mesh_per_dimension = canonical_run_metadata_uint64(
        run_metadata_json, "ic_mesh_per_dimension");
    provenance.lpt_order = canonical_run_metadata_int(
        run_metadata_json, "lpt_order");
    provenance.power_spectrum_sha256 = canonical_run_metadata_string(
        run_metadata_json, "power_spectrum_sha256");
    provenance.power_spectrum_redshift = canonical_run_metadata_real(
        run_metadata_json, "power_spectrum_redshift");
    provenance.power_spectrum_fidelity = canonical_run_metadata_string(
        run_metadata_json, "power_spectrum_fidelity");
    provenance.ic_amplitude_mode = canonical_run_metadata_string(
        run_metadata_json, "ic_amplitude_mode");
    provenance.ic_phase_pairing = canonical_run_metadata_string(
        run_metadata_json, "ic_phase_pairing");
    provenance.ic_lattice_convention = canonical_run_metadata_string(
        run_metadata_json, "ic_lattice_convention");
    provenance.ic_max_mode_per_axis = nullable_run_metadata_uint64(
        run_metadata_json, "ic_max_mode_per_axis");
    provenance.ic_effective_max_mode_per_axis = nullable_run_metadata_uint64(
        run_metadata_json, "ic_effective_max_mode_per_axis");

    if (ic_mode == "generate") {
        if (!generation_parameters_applied
            || (provenance_status
                    != "internal_generation_parameters_and_input_hash_applied"
                && provenance_status
                    != "internal_generation_parameters_applied_input_hash_unavailable")) {
            throw std::runtime_error(
                "Generated snapshot RunMetadataJson has inconsistent IC provenance status");
        }
        provenance.available = true;
    } else if (ic_mode == "snapshot") {
        if (generation_parameters_applied) {
            throw std::runtime_error(
                "Snapshot-sourced RunMetadataJson cannot claim that it applied generator parameters");
        }
        if (provenance_status
            == "external_snapshot_bytes_verified_upstream_generation_provenance_inherited") {
            provenance.available = true;
        } else if (provenance_status
                       == "external_snapshot_bytes_verified_upstream_generation_not_verified"
                   || provenance_status
                       == "external_snapshot_source_provenance_not_verified_input_hash_unavailable") {
            provenance.available = false;
        } else {
            throw std::runtime_error(
                "Snapshot-sourced RunMetadataJson has unsupported IC provenance status");
        }
    } else {
        throw std::runtime_error(
            "Snapshot RunMetadataJson has unsupported IC mode");
    }

    if (provenance.available) {
        if ((provenance.lpt_order != 1 && provenance.lpt_order != 2)
            || provenance.ic_mesh_per_dimension == 0
            || (provenance.ic_amplitude_mode != "gaussian"
                && provenance.ic_amplitude_mode != "fixed")
            || (provenance.ic_phase_pairing != "independent"
                && provenance.ic_phase_pairing != "pair_a"
                && provenance.ic_phase_pairing != "pair_b")
            || provenance.ic_lattice_convention != "corner"
            || !is_persisted_power_spectrum_fidelity(
                provenance.power_spectrum_fidelity)
            || provenance.power_spectrum_fidelity.empty()
            || !std::isfinite(provenance.power_spectrum_redshift)
            || provenance.power_spectrum_redshift <= -1.0
            || (!provenance.power_spectrum_sha256.empty()
                && !is_canonical_sha256(
                    provenance.power_spectrum_sha256))) {
            throw std::runtime_error(
                "Snapshot RunMetadataJson contains incomplete upstream generator provenance");
        }
    } else if (provenance.seed != 0
               || provenance.ic_mesh_per_dimension != 0
               || provenance.lpt_order != 0
               || !provenance.power_spectrum_sha256.empty()
               || provenance.power_spectrum_redshift != 0.0
               || !provenance.power_spectrum_fidelity.empty()
               || !provenance.ic_amplitude_mode.empty()
               || !provenance.ic_phase_pairing.empty()
               || !provenance.ic_lattice_convention.empty()) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson exposes unverified generator coordinates");
    }
    const auto requested = provenance.ic_max_mode_per_axis;
    const auto effective = provenance.ic_effective_max_mode_per_axis;
    if (provenance.available && !effective) {
        throw std::runtime_error(
            "Snapshot verified generator provenance requires effective IC support");
    }
    if (requested && (!effective || *requested != *effective)) {
        throw std::runtime_error("Snapshot requested and effective IC support disagree");
    }
    if (effective) {
        const auto n = canonical_run_metadata_uint64(
            run_metadata_json, "particles_per_dimension");
        if (!provenance.available || n == 0 || *effective == 0
            || *effective > (n - 1) / 2
            || (!requested && *effective != (n - 1) / 2)) {
            throw std::runtime_error("Snapshot IC support is inconsistent with its generator coordinates");
        }
    }
    return provenance;
}

void require_snapshot_initial_condition_match(
    const config::SimulationParameters& config,
    const SnapshotDescriptor& descriptor,
    std::string_view admitted_snapshot_sha256) {
    if (!is_canonical_sha256(admitted_snapshot_sha256)
        || descriptor.native_snapshot_object_sha256
            != admitted_snapshot_sha256) {
        throw std::runtime_error(
            "Snapshot descriptor and admitted exact IC object have different SHA-256 identities");
    }
    // Shared by serial, distributed and restart source re-admission. A native
    // container must not bypass the generated-IC input semantics merely by
    // hiding them behind snapshot mode. The origin hash binds declared input
    // bytes; it does not certify the external Boltzmann calculation.
    const auto& origin = descriptor.generation_provenance;
    if (!origin.available
        || !origin.ic_effective_max_mode_per_axis.has_value()
        || descriptor.power_spectrum_fidelity != "precision_boltzmann"
        || origin.power_spectrum_fidelity != "precision_boltzmann"
        || !is_canonical_sha256(origin.power_spectrum_sha256)) {
        throw std::runtime_error(
            "Snapshot IC requires retained precision_boltzmann generation "
            "provenance, effective Fourier support, and a canonical source power-spectrum SHA-256; "
            "regenerate from documented input rather than relabelling a snapshot");
    }
    if (descriptor.particle_count != config.num_particles()
        || descriptor.particles_per_dimension != config.get_box().N) {
        throw std::runtime_error(
            "Snapshot IC descriptor particle population disagrees with configured N^3");
    }

    const auto& cosmology = config.get_cosmology();
    if (!nearly_equal(descriptor.box_size_Mpc_h, config.get_box().L)
        || !nearly_equal(descriptor.omega_m, cosmology.omega_m)
        || !nearly_equal(
            descriptor.omega_lambda, cosmology.omega_lambda)
        || !nearly_equal(descriptor.omega_b, cosmology.omega_b)
        || !nearly_equal(descriptor.hubble_param, cosmology.h)
        || !nearly_equal(descriptor.sigma8, cosmology.sigma8)
        || !nearly_equal(descriptor.spectral_index_ns, cosmology.n_s)) {
        throw std::runtime_error(
            "Snapshot IC descriptor disagrees with the configured box or cosmology");
    }

    const core::Real expected_a = core::Real{1.0}
        / (core::Real{1.0} + config.get_time().z_start);
    if (!nearly_equal(descriptor.scale_factor, expected_a)) {
        throw std::runtime_error(
            "Snapshot IC scale factor does not match configured time.start_redshift");
    }

    const core::Real expected_particle_mass = config.particle_mass();
    if (!std::isfinite(expected_particle_mass)
        || expected_particle_mass <= 0.0) {
        throw std::runtime_error(
            "Configured particle mass is invalid for snapshot IC validation");
    }
    if (descriptor.uniform_mass
        && descriptor.uniform_particle_mass != expected_particle_mass) {
        throw std::runtime_error(
            "Snapshot IC uniform particle mass does not match configured cosmology and box resolution");
    }
}

SnapshotDescriptor read_snapshot_descriptor(const std::string& filename) {
    if (filename.empty()) {
        throw std::invalid_argument("Snapshot descriptor path must not be empty");
    }

    VerifiedSnapshotSource source(filename);
    return read_snapshot_descriptor(source);
}

SnapshotDescriptor read_snapshot_descriptor(
    const VerifiedSnapshotSource& source) {
    const std::string& descriptor_path = source.hdf5_read_path();

    // Bind descriptor metadata to one already-open regular-file object. The
    // retained descriptor survives pathname replacement, and its SHA-256 is
    // carried forward so payload ingestion can verify it reopened the same bytes.
    auto file = H5FileHandle::checked(
        H5Fopen(descriptor_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
        "open admitted snapshot descriptor");
    require_hard_link(file.get(), schema::GROUP_HEADER);
    require_hard_link(file.get(), schema::GROUP_PARAMETERS);
    require_hard_link(file.get(), schema::GROUP_CONFIG);

    auto header = H5GroupHandle::checked(
        H5Gopen2(file.get(), schema::GROUP_HEADER.c_str(), H5P_DEFAULT),
        "snapshot Header group");
    auto parameters = H5GroupHandle::checked(
        H5Gopen2(file.get(), schema::GROUP_PARAMETERS.c_str(), H5P_DEFAULT),
        "snapshot Parameters group");
    auto config = H5GroupHandle::checked(
        H5Gopen2(file.get(), schema::GROUP_CONFIG.c_str(), H5P_DEFAULT),
        "snapshot Config group");

    const std::string software_name = read_nonempty_string(
        config.get(),
        schema::ATTR_SOFTWARE_NAME,
        schema::VAL_SOFTWARE_NAME.size());
    if (std::string_view(software_name) != schema::VAL_SOFTWARE_NAME) {
        throw std::runtime_error(
            "Snapshot SoftwareName is not the HYOWON native identity");
    }
    const std::string snapshot_schema = read_nonempty_string(
        config.get(),
        schema::ATTR_SNAPSHOT_SCHEMA,
        schema::VAL_SNAPSHOT_SCHEMA.size());
    if (std::string_view(snapshot_schema) != schema::VAL_SNAPSHOT_SCHEMA) {
        throw std::runtime_error("Snapshot HYOWON schema is unsupported");
    }

    SnapshotDescriptor descriptor;
    descriptor.physics_fingerprint = read_nonempty_string(
        config.get(), schema::ATTR_PHYSICS_FINGERPRINT,
        MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES);
    descriptor.run_metadata_json = read_nonempty_string(
        config.get(), schema::ATTR_RUN_METADATA_JSON,
        MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES);

    const core::Real metadata_h = canonical_run_metadata_real(
        descriptor.run_metadata_json, "h");
    const core::Real metadata_omega_m = canonical_run_metadata_real(
        descriptor.run_metadata_json, "omega_m");
    const core::Real metadata_omega_lambda = canonical_run_metadata_real(
        descriptor.run_metadata_json, "omega_lambda");
    const core::Real metadata_omega_b = canonical_run_metadata_real(
        descriptor.run_metadata_json, "omega_b");
    const core::Real metadata_sigma8 = canonical_run_metadata_real(
        descriptor.run_metadata_json, "sigma8");
    const core::Real metadata_n_s = canonical_run_metadata_real(
        descriptor.run_metadata_json, "n_s");
    const core::Real metadata_box_size = canonical_run_metadata_real(
        descriptor.run_metadata_json, "box_size_Mpc_h");
    const std::uint64_t metadata_particles_per_dimension =
        canonical_run_metadata_uint64(
            descriptor.run_metadata_json, "particles_per_dimension");
    const std::uint64_t metadata_pm_mesh_per_dimension =
        canonical_run_metadata_uint64(
            descriptor.run_metadata_json, "pm_mesh_per_dimension");
    descriptor.generation_provenance = read_snapshot_generation_provenance(
        descriptor.run_metadata_json);
    const auto requested_support = canonical_run_metadata_string(
        descriptor.run_metadata_json, "ic_mode") == "generate"
        ? descriptor.generation_provenance.ic_max_mode_per_axis
        : std::nullopt;
    if (fingerprint_ic_support(descriptor.physics_fingerprint) != requested_support) {
        throw std::runtime_error("Snapshot IC support metadata disagrees with PhysicsFingerprint");
    }
    if (descriptor.generation_provenance.power_spectrum_fidelity.size()
        > schema::POWER_SPECTRUM_FIDELITY_MAX_BYTES) {
        throw std::runtime_error(
            "Snapshot RunMetadataJson PowerSpectrumFidelity exceeds native bound");
    }

    descriptor.scale_factor = static_cast<core::Real>(
        read_scalar<double>(header.get(), schema::ATTR_TIME, H5T_NATIVE_DOUBLE));
    descriptor.redshift = static_cast<core::Real>(
        read_scalar<double>(header.get(), schema::ATTR_REDSHIFT, H5T_NATIVE_DOUBLE));
    descriptor.box_size_Mpc_h = static_cast<core::Real>(
        read_scalar<double>(header.get(), schema::ATTR_BOX_SIZE, H5T_NATIVE_DOUBLE));
    descriptor.omega_m = static_cast<core::Real>(
        read_scalar<double>(header.get(), schema::ATTR_OMEGA0, H5T_NATIVE_DOUBLE));
    descriptor.omega_lambda = static_cast<core::Real>(
        read_scalar<double>(header.get(), schema::ATTR_OMEGA_LAMBDA, H5T_NATIVE_DOUBLE));
    descriptor.hubble_param = static_cast<core::Real>(
        read_scalar<double>(header.get(), schema::ATTR_HUBBLE_PARAM, H5T_NATIVE_DOUBLE));

    const auto count_this = read_array<std::uint32_t, 6>(
        header.get(), schema::ATTR_NUM_PART_THIS_FILE, H5T_NATIVE_UINT32);
    const auto count_total = read_array<std::uint32_t, 6>(
        header.get(), schema::ATTR_NUM_PART_TOTAL, H5T_NATIVE_UINT32);
    const auto mass_table = read_array<double, 6>(
        header.get(), schema::ATTR_MASS_TABLE, H5T_NATIVE_DOUBLE);
    const std::int32_t file_count = read_scalar<std::int32_t>(
        header.get(), schema::ATTR_NUM_FILES, H5T_NATIVE_INT32);
    if (file_count != 1) {
        throw std::runtime_error("Snapshot descriptor requires one complete native file");
    }
    for (std::size_t type = 0; type < count_this.size(); ++type) {
        if (count_this[type] != count_total[type]) {
            throw std::runtime_error("Snapshot particle count attributes disagree");
        }
        if (type != 1 && count_total[type] != 0) {
            throw std::runtime_error(
                "Snapshot descriptor supports only the effective collisionless PartType1 cb population");
        }
        if (!std::isfinite(mass_table[type]) || mass_table[type] < 0.0) {
            throw std::runtime_error("Snapshot mass table is invalid");
        }
    }
    descriptor.particle_count = count_total[1];
    descriptor.uniform_mass = mass_table[1] > 0.0;
    descriptor.uniform_particle_mass = static_cast<core::Real>(mass_table[1]);

    const double unit_length = read_scalar<double>(
        header.get(), schema::ATTR_UNIT_LENGTH_IN_CM, H5T_NATIVE_DOUBLE);
    const double unit_mass = read_scalar<double>(
        header.get(), schema::ATTR_UNIT_MASS_IN_G, H5T_NATIVE_DOUBLE);
    const double unit_velocity = read_scalar<double>(
        header.get(), schema::ATTR_UNIT_VELOCITY_IN_CM_PER_S, H5T_NATIVE_DOUBLE);
    if (!nearly_equal(unit_length, cosmology::units::UnitLength_in_cm)
        || !nearly_equal(unit_mass, cosmology::units::UnitMass_in_g)
        || !nearly_equal(unit_velocity, cosmology::units::UnitVelocity_in_cm_s)) {
        throw std::runtime_error("Snapshot descriptor unit system is not native");
    }

    descriptor.ic_seed = read_scalar<std::uint64_t>(
        parameters.get(), schema::ATTR_SEED, H5T_NATIVE_UINT64);
    descriptor.particles_per_dimension = read_scalar<std::uint32_t>(
        parameters.get(), schema::ATTR_PARTICLES_PER_DIM, H5T_NATIVE_UINT32);
    descriptor.pm_mesh_per_dimension = read_scalar<std::uint32_t>(
        parameters.get(), schema::ATTR_PM_MESH_PER_DIM, H5T_NATIVE_UINT32);
    descriptor.ic_mesh_per_dimension = read_scalar<std::uint32_t>(
        parameters.get(), schema::ATTR_IC_MESH_PER_DIM, H5T_NATIVE_UINT32);
    if (descriptor.ic_seed != descriptor.generation_provenance.seed
        || descriptor.ic_mesh_per_dimension
            != descriptor.generation_provenance.ic_mesh_per_dimension) {
        throw std::runtime_error(
            "Snapshot IC generation attributes disagree with RunMetadataJson");
    }
    descriptor.softening_comoving_Mpc_h = static_cast<core::Real>(
        read_scalar<double>(
            parameters.get(), schema::ATTR_SOFTENING_COMOVING, H5T_NATIVE_DOUBLE));
    descriptor.sigma8 = static_cast<core::Real>(
        read_scalar<double>(parameters.get(), schema::ATTR_SIGMA8, H5T_NATIVE_DOUBLE));
    descriptor.spectral_index_ns = static_cast<core::Real>(
        read_scalar<double>(parameters.get(), schema::ATTR_N_S, H5T_NATIVE_DOUBLE));

    descriptor.omega_b = static_cast<core::Real>(
        read_scalar<double>(
            parameters.get(), schema::ATTR_OMEGA_B, H5T_NATIVE_DOUBLE));
    if (descriptor.omega_b != metadata_omega_b) {
        throw std::runtime_error(
            "Snapshot OmegaB attribute disagrees with RunMetadataJson");
    }

    descriptor.lpt_order = read_scalar<std::int32_t>(
        parameters.get(), schema::ATTR_LPT_ORDER, H5T_NATIVE_INT32);
    if (descriptor.lpt_order != descriptor.generation_provenance.lpt_order) {
        throw std::runtime_error(
            "Snapshot LptOrder attribute disagrees with RunMetadataJson");
    }

    descriptor.ic_amplitude_mode = read_required_string(
        parameters.get(), schema::ATTR_IC_AMPLITUDE_MODE, 8U);
    descriptor.ic_phase_pairing = read_required_string(
        parameters.get(), schema::ATTR_IC_PHASE_PAIRING, 11U);
    if (descriptor.ic_amplitude_mode
            != descriptor.generation_provenance.ic_amplitude_mode
        || descriptor.ic_phase_pairing
            != descriptor.generation_provenance.ic_phase_pairing) {
        throw std::runtime_error(
            "Snapshot IC realization attributes disagree with RunMetadataJson");
    }

    descriptor.power_spectrum_fidelity = read_required_string(
        parameters.get(),
        schema::ATTR_POWER_SPECTRUM_FIDELITY,
        schema::POWER_SPECTRUM_FIDELITY_MAX_BYTES);
    if (descriptor.power_spectrum_fidelity
        != descriptor.generation_provenance.power_spectrum_fidelity) {
        throw std::runtime_error(
            "Snapshot PowerSpectrumFidelity attribute disagrees with RunMetadataJson");
    }

    if (descriptor.hubble_param != metadata_h
        || descriptor.omega_m != metadata_omega_m
        || descriptor.omega_lambda != metadata_omega_lambda
        || descriptor.omega_b != metadata_omega_b
        || descriptor.sigma8 != metadata_sigma8
        || descriptor.spectral_index_ns != metadata_n_s
        || descriptor.box_size_Mpc_h != metadata_box_size
        || descriptor.particles_per_dimension
            != metadata_particles_per_dimension
        || descriptor.pm_mesh_per_dimension
            != metadata_pm_mesh_per_dimension) {
        throw std::runtime_error(
            "Snapshot native physics attributes disagree with RunMetadataJson");
    }

    for (std::size_t index = 0;
         index < schema::NATIVE_SNAPSHOT_SEMANTIC_BINDINGS.size();
         ++index) {
        const auto& binding = schema::NATIVE_SNAPSHOT_SEMANTIC_BINDINGS[index];
        descriptor.semantic_binding_values[index] = read_nonempty_string(
            config.get(),
            std::string(binding.attribute),
            binding.value.size());
        if (descriptor.semantic_binding_values[index] != binding.value) {
            throw std::runtime_error(
                "Snapshot descriptor semantic binding is unsupported: "
                + std::string(binding.attribute));
        }
    }

    require_positive(descriptor.scale_factor, "scale_factor");
    require_finite(descriptor.redshift, "redshift");
    if (!nearly_equal(
            descriptor.redshift,
            core::Real{1.0} / descriptor.scale_factor - core::Real{1.0})) {
        throw std::runtime_error("Snapshot descriptor redshift disagrees with scale factor");
    }
    require_positive(descriptor.box_size_Mpc_h, "box_size_Mpc_h");
    require_positive(descriptor.omega_m, "omega_m");
    require_finite(descriptor.omega_lambda, "omega_lambda");
    if (descriptor.omega_lambda < 0.0) {
        throw std::runtime_error(
            "Snapshot descriptor OmegaLambda must be non-negative");
    }
    if (!cosmology::has_flat_matter_lambda_closure(
            descriptor.omega_m,
            descriptor.omega_lambda)) {
        throw std::runtime_error(
            "Snapshot descriptor requires flat matter-plus-Lambda cosmology");
    }
    require_positive(descriptor.hubble_param, "hubble_param");
    // Pure PM has no Plummer short-range softening. Native PM snapshots carry
    // zero in this field; TreePM configurations guarantee a positive value at
    // construction time and persist it in the physics fingerprint.
    require_nonnegative(
        descriptor.softening_comoving_Mpc_h,
        "softening_comoving_Mpc_h");
    require_finite(descriptor.sigma8, "sigma8");
    if (descriptor.sigma8 < 0.0) {
        throw std::runtime_error(
            "Snapshot descriptor sigma8 must be non-negative");
    }
    require_finite(descriptor.spectral_index_ns, "spectral_index_ns");
    require_nonnegative(descriptor.omega_b, "omega_b");
    if (descriptor.omega_b > descriptor.omega_m) {
        throw std::runtime_error(
            "Snapshot descriptor omega_b must not exceed omega_m");
    }
    if (descriptor.lpt_order != 0
        && descriptor.lpt_order != 1
        && descriptor.lpt_order != 2) {
        throw std::runtime_error(
            "Snapshot descriptor LptOrder must be 0, 1, or 2");
    }
    // LptOrder=0 is the native sentinel when upstream generation provenance is
    // unavailable. Generated snapshots and snapshot-sourced evolutions that
    // inherited verified generator coordinates retain 1LPT/2LPT plus their
    // realization controls; ic_generation_parameters_applied in RunMetadataJson
    // still distinguishes whether this particular run performed generation.
    if (descriptor.lpt_order == 0) {
        if (!descriptor.ic_amplitude_mode.empty()
            || !descriptor.ic_phase_pairing.empty()) {
            throw std::runtime_error(
                "Snapshot descriptor without generator provenance has active IC realization controls");
        }
    } else if (descriptor.ic_amplitude_mode.empty()
               || descriptor.ic_phase_pairing.empty()) {
        throw std::runtime_error(
            "Snapshot descriptor with generator provenance is missing realization controls");
    }
    if (!is_persisted_power_spectrum_fidelity(descriptor.power_spectrum_fidelity)) {
        throw std::runtime_error(
            "Snapshot descriptor PowerSpectrumFidelity is unsupported");
    }
    if (descriptor.particle_count == 0
        || descriptor.particles_per_dimension == 0
        || descriptor.pm_mesh_per_dimension == 0) {
        throw std::runtime_error(
            "Snapshot descriptor particle and PM resolution fields must be positive");
    }
    if (descriptor.lpt_order != 0 && descriptor.ic_mesh_per_dimension == 0) {
        throw std::runtime_error(
            "Generated-IC descriptor requires a positive IC mesh dimension");
    }
    if (descriptor.lpt_order != 0
        && descriptor.ic_mesh_per_dimension
            % descriptor.particles_per_dimension != 0) {
        throw std::runtime_error(
            "Generated-IC descriptor requires its IC mesh dimension to be an integer multiple of the particle dimension");
    }
    if (descriptor.lpt_order == 2
        && descriptor.ic_mesh_per_dimension / 2
            < descriptor.particles_per_dimension) {
        throw std::runtime_error(
            "Generated 2LPT descriptor requires its IC mesh dimension to be at least twice the particle dimension");
    }
    if (descriptor.particles_per_dimension
        > std::numeric_limits<std::uint64_t>::max()
            / descriptor.particles_per_dimension
        || descriptor.particles_per_dimension * descriptor.particles_per_dimension
            > std::numeric_limits<std::uint64_t>::max()
                / descriptor.particles_per_dimension) {
        throw std::overflow_error("Snapshot descriptor N^3 overflows uint64");
    }
    const std::uint64_t expected_count =
        descriptor.particles_per_dimension
        * descriptor.particles_per_dimension
        * descriptor.particles_per_dimension;
    if (expected_count != descriptor.particle_count) {
        throw std::runtime_error("Snapshot descriptor particle count disagrees with N^3");
    }
    if (descriptor.uniform_mass) {
        require_positive(descriptor.uniform_particle_mass, "uniform_particle_mass");
    }

    // Close every HDF5 alias before re-hashing the retained regular-file object;
    // this also avoids shared-offset interference on macOS /dev/fd aliases.
    config.reset();
    parameters.reset();
    header.reset();
    file.reset();
    source.verify_unchanged();
    descriptor.native_snapshot_object_sha256 = source.sha256();
    return descriptor;
}

void require_field_comparison_match(
    const SnapshotDescriptor& reference,
    const SnapshotDescriptor& candidate) {
    constexpr const char* product = "Field cross-correlation";
    require_semantic_bindings(reference, candidate, product);
    require_equal(
        reference.box_size_Mpc_h,
        candidate.box_size_Mpc_h,
        product,
        "box_size_Mpc_h");
}

void require_pair_link_match(
    const SnapshotDescriptor& earlier,
    const SnapshotDescriptor& later) {
    constexpr const char* product = "Stable-ID pair link";
    require_semantic_bindings(earlier, later, product);
    require_equal(
        earlier.physics_fingerprint,
        later.physics_fingerprint,
        product,
        "physics_fingerprint");
    require_equal(
        earlier.box_size_Mpc_h,
        later.box_size_Mpc_h,
        product,
        "box_size_Mpc_h");
    require_equal(
        earlier.particle_count,
        later.particle_count,
        product,
        "particle_count");
    require_equal(
        earlier.particles_per_dimension,
        later.particles_per_dimension,
        product,
        "particles_per_dimension");
    require_equal(
        earlier.uniform_mass,
        later.uniform_mass,
        product,
        "mass_representation");
    if (earlier.uniform_mass) {
        require_equal(
            earlier.uniform_particle_mass,
            later.uniform_particle_mass,
            product,
            "uniform_particle_mass");
    }
    if (!(earlier.scale_factor < later.scale_factor)) {
        comparison_mismatch(product, "strict_temporal_order");
    }
}

} // namespace cosmo_nbody::io
