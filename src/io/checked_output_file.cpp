#include "cosmo_nbody/io/checked_output_file.hpp"

#include <stdexcept>
#include <string>
#include <exception>

namespace cosmo_nbody::io {
namespace {

void require_output_identity(
    const std::filesystem::path& path,
    std::string_view role) {
    if (path.empty() || path.filename().empty()) {
        throw std::invalid_argument(
            "Checked output path must name a file");
    }
    if (role.empty()) {
        throw std::invalid_argument(
            "Checked output role must not be empty");
    }
}

std::runtime_error output_error(
    std::string_view operation,
    const std::filesystem::path& path,
    std::string_view role) {
    std::string message =
        "Failed while " + std::string(operation) + " " + std::string(role)
        + ": " + path.string();
    return std::runtime_error(std::move(message));
}

} // namespace

CheckedOutputFile::CheckedOutputFile(
    const std::filesystem::path& path,
    std::string_view role)
    : requested_path_(path),
      role_(role),
      publication_(
          path,
          std::string(role),
          DurableFilePublicationPolicy::RequireAbsent),
      output_(
          publication_.staging_path(),
          std::ios::binary | std::ios::trunc) {
    require_output_identity(requested_path_, role_);
    if (!output_) {
        throw output_error("creating staged", path, role);
    }
    output_.precision(17);
}

void CheckedOutputFile::publish(
    const std::filesystem::path& expected_path,
    std::string_view expected_role) {
    require_output_identity(expected_path, expected_role);
    if (expected_path != requested_path_ || expected_role != role_) {
        throw std::logic_error(
            "Checked output close identity disagrees with its open transaction");
    }
    if (published_) {
        throw std::logic_error(
            "Checked output transaction was already published: "
            + requested_path_.string());
    }
    if (!output_.is_open()) {
        throw std::logic_error(
            "Checked output stream is not open for " + role_
            + ": " + requested_path_.string());
    }

    output_.flush();
    const bool flush_failed = output_.fail();
    output_.close();
    const bool close_failed = output_.fail();
    if (flush_failed || close_failed) {
        try {
            publication_.discard_unpublished();
        } catch (...) {
            std::throw_with_nested(output_error(
                "cleaning failed staged", requested_path_, role_));
        }
        throw output_error("writing staged", requested_path_, role_);
    }

    publication_.publish_nonempty();
    published_ = true;
}

CheckedOutputFile open_checked_output_file(
    const std::filesystem::path& path,
    std::string_view role) {
    require_output_identity(path, role);
    return CheckedOutputFile(path, role);
}

void close_checked_output_file(
    CheckedOutputFile& output,
    const std::filesystem::path& path,
    std::string_view role) {
    output.publish(path, role);
}

} // namespace cosmo_nbody::io
