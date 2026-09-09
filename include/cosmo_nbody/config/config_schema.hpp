// Typed public TOML keys used by ConfigLoader and scalar overrides.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <string_view>

namespace cosmo_nbody {
namespace config {

enum class ConfigValueKind {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    Real,
    String,
    RealArray,
};

struct ConfigKeySpec {
    ConfigValueKind kind{ConfigValueKind::String};

    bool scalar_override_allowed() const noexcept {
        return kind != ConfigValueKind::RealArray;
    }
};

class ConfigSchema final {
public:
    using KeyMap = std::map<std::string, ConfigKeySpec, std::less<>>;
    using SectionMap = std::map<std::string, KeyMap, std::less<>>;

    static const ConfigSchema& instance();

    bool has_section(std::string_view section) const;
    bool has_key(std::string_view section, std::string_view key) const;
    const ConfigKeySpec* find(
        std::string_view section,
        std::string_view key) const noexcept;

private:
    ConfigSchema();

    SectionMap sections_;
};

} // namespace config
} // namespace cosmo_nbody
