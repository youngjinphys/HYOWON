#pragma once

#include <filesystem>
#include <functional>
#include <string_view>

namespace cosmo_nbody::io {

using DurableTextStagingValidator = std::function<void(
    const std::filesystem::path& staged_path)>;

// Durably replace a text artifact through a unique same-directory stage.
void write_text_durable_atomic(
    const std::filesystem::path& path,
    std::string_view content,
    std::string_view role);

// Validate the completed stage, then publish with atomic no-replace semantics;
// unsupported filesystems fail instead of falling back to a racy replacement.
void write_text_durable_atomic_validated(
    const std::filesystem::path& path,
    std::string_view content,
    std::string_view role,
    const DurableTextStagingValidator& validator);

} // namespace cosmo_nbody::io
