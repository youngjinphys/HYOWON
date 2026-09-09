#include "cosmo_nbody/io/restart_lineage.hpp"

#include "cosmo_nbody/io/content_hash.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace cosmo_nbody::io {
namespace {

std::optional<std::string>& parent_manifest_sha256() {
    static std::optional<std::string> value;
    return value;
}

} // namespace

void clear_process_restart_parent() noexcept {
    parent_manifest_sha256().reset();
}

void set_process_restart_parent(std::string manifest_sha256) {
    if (!is_canonical_sha256(manifest_sha256)) {
        throw std::invalid_argument(
            "Restart parent manifest identity must be a canonical SHA-256");
    }
    parent_manifest_sha256() = std::move(manifest_sha256);
}

std::optional<std::string> process_restart_parent_manifest_sha256() {
    return parent_manifest_sha256();
}

} // namespace cosmo_nbody::io
