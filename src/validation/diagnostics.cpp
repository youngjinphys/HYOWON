#include "cosmo_nbody/validation/diagnostics.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cosmo_nbody {
namespace validation {

namespace {

constexpr const char* diagnostic_interpretation =
    "runtime_measurements_only_require_external_observable_specific_convergence";

std::string json_escape(const std::string& value) {
    static constexpr char hexadecimal[] = "0123456789abcdef";

    std::string out;
    out.reserve(value.size() + 2);
    for (const char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: {
                const unsigned char byte = static_cast<unsigned char>(c);
                if (byte < 0x20U) {
                    out += "\\u00";
                    out += hexadecimal[(byte >> 4U) & 0x0fU];
                    out += hexadecimal[byte & 0x0fU];
                } else {
                    out += c;
                }
                break;
            }
        }
    }
    return out;
}

void write_json_real(std::ostringstream& out, core::Real value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
}

} // namespace

DiagnosticReport::DiagnosticReport(io::RunMetadata metadata)
    : metadata_(std::move(metadata)) {}

void DiagnosticReport::add_check(DiagnosticCheck check) {
    if (check.name.empty()) {
        throw std::invalid_argument(
            "Diagnostic measurement name must not be empty");
    }
    checks_.push_back(std::move(check));
}

std::string DiagnosticReport::summary_text() const {
    std::ostringstream out;
    out.precision(6);
    out << "Diagnostic report: "
        << checks_.size() << " measurements\n"
        << "INTERPRETATION verdict_semantics=false"
        << " scientific_accuracy_certificate=false"
        << " scope=" << diagnostic_interpretation << "\n";
    for (const auto& check : checks_) {
        out << "MEASURE " << check.name << " value=" << check.value;
        if (!check.units.empty()) out << " " << check.units;
        if (!check.message.empty()) out << " - " << check.message;
        out << "\n";
    }
    return out.str();
}

std::string DiagnosticReport::to_json() const {
    std::ostringstream out;
    out.precision(17);
    out << "{\"product_kind\":\"diagnostic_report\""
        << ",\"verdict_semantics\":false"
        << ",\"scientific_accuracy_certificate\":false"
        << ",\"interpretation\":\""
        << diagnostic_interpretation << "\""
        << ",\"metadata\":" << metadata_.to_json()
        << ",\"checks\":[";
    for (std::size_t i = 0; i < checks_.size(); ++i) {
        const auto& check = checks_[i];
        if (i != 0) out << ",";
        out << "{";
        out << "\"name\":\"" << json_escape(check.name) << "\",";
        out << "\"value\":";
        write_json_real(out, check.value);
        out << ",\"units\":\"" << json_escape(check.units) << "\",";
        out << "\"message\":\"" << json_escape(check.message) << "\"";
        out << "}";
    }
    out << "]}";
    return out.str();
}

} // namespace validation
} // namespace cosmo_nbody
