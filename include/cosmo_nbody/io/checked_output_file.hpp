#pragma once

#include "cosmo_nbody/io/durable_file_publication.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>

namespace cosmo_nbody::io {

// Streaming create-only publication for analyzer text products. Each instance
// owns a unique staging file; publish() durably exposes an absent final
// pathname only after the stream closes. The analyzer's exclusively created
// execution directory owns the multi-product namespace.
class CheckedOutputFile {
public:
    CheckedOutputFile(
        const std::filesystem::path& path,
        std::string_view role);
    ~CheckedOutputFile() noexcept = default;

    CheckedOutputFile(const CheckedOutputFile&) = delete;
    CheckedOutputFile& operator=(const CheckedOutputFile&) = delete;
    CheckedOutputFile(CheckedOutputFile&&) = delete;
    CheckedOutputFile& operator=(CheckedOutputFile&&) = delete;

    operator std::ofstream&() noexcept { return output_; }
    operator const std::ofstream&() const noexcept { return output_; }

    template <typename Value>
    CheckedOutputFile& operator<<(Value&& value) {
        output_ << std::forward<Value>(value);
        return *this;
    }

    std::streamsize precision(std::streamsize value) {
        return output_.precision(value);
    }

    void publish(
        const std::filesystem::path& expected_path,
        std::string_view expected_role);

private:
    std::filesystem::path requested_path_;
    std::string role_;
    DurableFilePublication publication_;
    std::ofstream output_;
    bool published_{false};
};

CheckedOutputFile open_checked_output_file(
    const std::filesystem::path& path,
    std::string_view role);

void close_checked_output_file(
    CheckedOutputFile& output,
    const std::filesystem::path& path,
    std::string_view role);

} // namespace cosmo_nbody::io
