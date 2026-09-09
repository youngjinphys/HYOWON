#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

namespace cosmo_nbody::runtime {

// Admit scratch directory before file creation. Explicit paths may not traverse
// symlinks; platform-selected temporary directories are canonicalized.
std::filesystem::path prepare_scratch_directory(
    const std::string& directory,
    std::string_view label);

// Reserve real backing before publishing a writable mapping so ENOSPC fails here
// rather than on a later mapped write.
void reserve_scratch_file_space(
    int fd,
    std::size_t bytes,
    std::string_view label);

// Configure mutable file-backed mapping policy. Linux random-access advice avoids
// speculative readahead/cache amplification; rejection is fail-closed.
void advise_mutable_scratch_mapping(
    void* mapping,
    std::size_t bytes,
    std::string_view label);

} // namespace cosmo_nbody::runtime
