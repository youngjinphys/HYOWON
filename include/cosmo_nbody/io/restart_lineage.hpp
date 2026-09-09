#pragma once

#include <optional>
#include <string>

namespace cosmo_nbody::io {

// Process-local trajectory provenance used when writing direct-simulation
// metadata. Restart checkpoint integrity is checked by RestartCheckpointIO;
// this state only remembers the verified parent manifest identity.
void clear_process_restart_parent() noexcept;
void set_process_restart_parent(std::string manifest_sha256);
std::optional<std::string> process_restart_parent_manifest_sha256();

class RestartLineageScope {
public:
    RestartLineageScope() noexcept {
        clear_process_restart_parent();
    }
    ~RestartLineageScope() noexcept {
        clear_process_restart_parent();
    }

    RestartLineageScope(const RestartLineageScope&) = delete;
    RestartLineageScope& operator=(const RestartLineageScope&) = delete;
    RestartLineageScope(RestartLineageScope&&) = delete;
    RestartLineageScope& operator=(RestartLineageScope&&) = delete;
};

} // namespace cosmo_nbody::io
