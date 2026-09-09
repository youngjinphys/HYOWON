#include "cosmo_nbody/config/config_schema.hpp"

namespace cosmo_nbody {
namespace config {

const ConfigSchema& ConfigSchema::instance() {
    static const ConfigSchema schema;
    return schema;
}

ConfigSchema::ConfigSchema() {
#define COSMO_CONFIG_KEY(section_name, key_name, kind_name) \
    sections_[section_name].emplace( \
        key_name, ConfigKeySpec{ConfigValueKind::kind_name});
#include "cosmo_nbody/config/config_keys.def"
#undef COSMO_CONFIG_KEY
}

bool ConfigSchema::has_section(std::string_view section) const {
    return sections_.find(section) != sections_.end();
}

bool ConfigSchema::has_key(
    std::string_view section,
    std::string_view key) const {
    return find(section, key) != nullptr;
}

const ConfigKeySpec* ConfigSchema::find(
    std::string_view section,
    std::string_view key) const noexcept {
    const auto section_it = sections_.find(section);
    if (section_it == sections_.end()) return nullptr;
    const auto key_it = section_it->second.find(key);
    return key_it == section_it->second.end() ? nullptr : &key_it->second;
}

} // namespace config
} // namespace cosmo_nbody
