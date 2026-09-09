// Diagnostic measurements carry no pass/fail threshold; scientific
// interpretation belongs to external convergence analysis.
#pragma once

#include "cosmo_nbody/io/metadata.hpp"
#include "cosmo_nbody/core/types.hpp"
#include <string>
#include <vector>

namespace cosmo_nbody {
namespace validation {

struct DiagnosticCheck {
    std::string name;
    core::Real value{0.0};
    std::string units;
    std::string message;
};

class DiagnosticReport {
public:
    explicit DiagnosticReport(io::RunMetadata metadata);

    void add_check(DiagnosticCheck check);

    std::string summary_text() const;
    std::string to_json() const;

private:
    io::RunMetadata metadata_;
    std::vector<DiagnosticCheck> checks_;
};

} // namespace validation
} // namespace cosmo_nbody
