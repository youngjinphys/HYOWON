#include "cosmo_nbody/io/content_hash.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cosmo_nbody::io {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

} // namespace

struct Sha256Accumulator::Impl {
public:
    void update(std::span<const std::byte> bytes) {
        if (finalized_) {
            throw std::logic_error("SHA-256 update after finalization");
        }
        if (bytes.size()
            > (std::numeric_limits<std::uint64_t>::max() - total_bytes_)) {
            throw std::overflow_error("SHA-256 input length exceeds uint64 byte range");
        }
        total_bytes_ += static_cast<std::uint64_t>(bytes.size());

        for (const std::byte value : bytes) {
            buffer_[buffer_size_++] = std::to_integer<std::uint8_t>(value);
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                buffer_size_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finalize() {
        if (finalized_) {
            throw std::logic_error("SHA-256 finalized more than once");
        }
        finalized_ = true;
        if (total_bytes_ > std::numeric_limits<std::uint64_t>::max() / 8U) {
            throw std::overflow_error("SHA-256 bit length exceeds uint64 range");
        }
        const std::uint64_t bit_length = total_bytes_ * 8U;

        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56U) {
            while (buffer_size_ < buffer_.size()) buffer_[buffer_size_++] = 0U;
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56U) buffer_[buffer_size_++] = 0U;
        for (int shift = 56; shift >= 0; shift -= 8) {
            buffer_[buffer_size_++] = static_cast<std::uint8_t>(
                (bit_length >> static_cast<unsigned int>(shift)) & 0xffU);
        }
        transform(buffer_.data());
        buffer_size_ = 0;

        std::array<std::uint8_t, 32> digest{};
        for (std::size_t word = 0; word < state_.size(); ++word) {
            digest[4U * word] = static_cast<std::uint8_t>(state_[word] >> 24U);
            digest[4U * word + 1U] = static_cast<std::uint8_t>(state_[word] >> 16U);
            digest[4U * word + 2U] = static_cast<std::uint8_t>(state_[word] >> 8U);
            digest[4U * word + 3U] = static_cast<std::uint8_t>(state_[word]);
        }
        return digest;
    }

private:
    static std::uint32_t choose(
        std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
        return (x & y) ^ (~x & z);
    }

    static std::uint32_t majority(
        std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    static std::uint32_t big_sigma0(std::uint32_t value) noexcept {
        return std::rotr(value, 2) ^ std::rotr(value, 13) ^ std::rotr(value, 22);
    }

    static std::uint32_t big_sigma1(std::uint32_t value) noexcept {
        return std::rotr(value, 6) ^ std::rotr(value, 11) ^ std::rotr(value, 25);
    }

    static std::uint32_t small_sigma0(std::uint32_t value) noexcept {
        return std::rotr(value, 7) ^ std::rotr(value, 18) ^ (value >> 3U);
    }

    static std::uint32_t small_sigma1(std::uint32_t value) noexcept {
        return std::rotr(value, 17) ^ std::rotr(value, 19) ^ (value >> 10U);
    }

    void transform(const std::uint8_t* block) noexcept {
        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const std::size_t offset = 4U * index;
            schedule[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24U)
                | (static_cast<std::uint32_t>(block[offset + 1U]) << 16U)
                | (static_cast<std::uint32_t>(block[offset + 2U]) << 8U)
                | static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t index = 16U; index < schedule.size(); ++index) {
            schedule[index] = small_sigma1(schedule[index - 2U])
                + schedule[index - 7U]
                + small_sigma0(schedule[index - 15U])
                + schedule[index - 16U];
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (std::size_t index = 0; index < schedule.size(); ++index) {
            const std::uint32_t temp1 = h + big_sigma1(e) + choose(e, f, g)
                + kRoundConstants[index] + schedule[index];
            const std::uint32_t temp2 = big_sigma0(a) + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_{0};
    std::uint64_t total_bytes_{0};
    bool finalized_{false};
};

namespace {

std::string hex_digest(const std::array<std::uint8_t, 32>& digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return out.str();
}

void encode_little_endian_u64(
    std::uint64_t value,
    std::byte* output) noexcept {
    for (std::size_t offset = 0; offset < sizeof(value); ++offset) {
        output[offset] = static_cast<std::byte>(
            (value >> (8U * offset)) & 0xffU);
    }
}

#if !defined(_WIN32)
class ScopedFileDescriptor {
public:
    explicit ScopedFileDescriptor(int descriptor) noexcept
        : descriptor_(descriptor) {}

    ~ScopedFileDescriptor() {
        if (descriptor_ >= 0) (void)::close(descriptor_);
    }

    ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
    ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

    int get() const noexcept { return descriptor_; }

private:
    int descriptor_{-1};
};

std::runtime_error file_hash_system_error(
    const char* operation,
    const std::filesystem::path& path,
    int error_number) {
    return std::runtime_error(
        std::string(operation) + ": " + path.string() + ": "
        + std::error_code(error_number, std::generic_category()).message());
}

bool same_file_identity(
    const struct stat& lhs,
    const struct stat& rhs) noexcept {
    const bool common = lhs.st_dev == rhs.st_dev
        && lhs.st_ino == rhs.st_ino
        && lhs.st_mode == rhs.st_mode
        && lhs.st_size == rhs.st_size;
#if defined(__APPLE__)
    return common
        && lhs.st_mtimespec.tv_sec == rhs.st_mtimespec.tv_sec
        && lhs.st_mtimespec.tv_nsec == rhs.st_mtimespec.tv_nsec
        && lhs.st_ctimespec.tv_sec == rhs.st_ctimespec.tv_sec
        && lhs.st_ctimespec.tv_nsec == rhs.st_ctimespec.tv_nsec;
#else
    return common
        && lhs.st_mtim.tv_sec == rhs.st_mtim.tv_sec
        && lhs.st_mtim.tv_nsec == rhs.st_mtim.tv_nsec
        && lhs.st_ctim.tv_sec == rhs.st_ctim.tv_sec
        && lhs.st_ctim.tv_nsec == rhs.st_ctim.tv_nsec;
#endif
}
#endif

} // namespace

Sha256Accumulator::Sha256Accumulator()
    : impl_(std::make_unique<Impl>()) {}

Sha256Accumulator::~Sha256Accumulator() = default;
Sha256Accumulator::Sha256Accumulator(Sha256Accumulator&&) noexcept = default;
Sha256Accumulator& Sha256Accumulator::operator=(
    Sha256Accumulator&&) noexcept = default;

void Sha256Accumulator::update(std::span<const std::byte> bytes) {
    if (!impl_) {
        throw std::logic_error("SHA-256 accumulator is not initialized");
    }
    impl_->update(bytes);
}

void Sha256Accumulator::update_canonical_uint64(
    std::span<const std::uint64_t> values) {
    constexpr std::size_t values_per_chunk = 4096U;
    std::array<std::byte,
               values_per_chunk * sizeof(std::uint64_t)> encoded{};
    for (std::size_t begin = 0; begin < values.size();
         begin += values_per_chunk) {
        const std::size_t count = std::min(
            values_per_chunk, values.size() - begin);
        for (std::size_t index = 0; index < count; ++index) {
            encode_little_endian_u64(
                values[begin + index],
                encoded.data() + index * sizeof(std::uint64_t));
        }
        update(std::span(
            encoded.data(), count * sizeof(std::uint64_t)));
    }
}

void Sha256Accumulator::update_canonical_real(
    std::span<const core::Real> values) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    constexpr std::size_t values_per_chunk = 4096U;
    std::array<std::byte,
               values_per_chunk * sizeof(std::uint64_t)> encoded{};
    for (std::size_t begin = 0; begin < values.size();
         begin += values_per_chunk) {
        const std::size_t count = std::min(
            values_per_chunk, values.size() - begin);
        for (std::size_t index = 0; index < count; ++index) {
            const core::Real value = values[begin + index];
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Canonical SHA-256 Real input must be finite");
            }
            const double canonical = value == core::Real{0.0}
                ? 0.0
                : static_cast<double>(value);
            std::uint64_t bits = 0;
            std::memcpy(&bits, &canonical, sizeof(bits));
            encode_little_endian_u64(
                bits,
                encoded.data() + index * sizeof(std::uint64_t));
        }
        update(std::span(
            encoded.data(), count * sizeof(std::uint64_t)));
    }
}

std::string Sha256Accumulator::finish_hex() {
    if (!impl_) {
        throw std::logic_error("SHA-256 accumulator is not initialized");
    }
    return hex_digest(impl_->finalize());
}

std::string sha256_bytes(std::span<const std::byte> bytes) {
    Sha256Accumulator hash;
    hash.update(bytes);
    return hash.finish_hex();
}

std::string sha256_file(const std::filesystem::path& path) {
#if !defined(_WIN32)
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int descriptor_value = ::open(path.c_str(), flags);
    if (descriptor_value < 0) {
        throw file_hash_system_error(
            "Could not open file for SHA-256 hashing", path, errno);
    }
    ScopedFileDescriptor descriptor(descriptor_value);

    struct stat before{};
    if (::fstat(descriptor.get(), &before) != 0) {
        throw file_hash_system_error(
            "Could not inspect open file before SHA-256 hashing", path, errno);
    }
    if (!S_ISREG(before.st_mode)) {
        throw std::runtime_error(
            "SHA-256 input must resolve to a regular file: " + path.string());
    }

    Sha256Accumulator hash;
    std::array<std::byte, 64U * 1024U> buffer{};
    while (true) {
        const ssize_t count = ::read(
            descriptor.get(), buffer.data(), buffer.size());
        if (count == 0) break;
        if (count < 0) {
            if (errno == EINTR) continue;
            throw file_hash_system_error(
                "I/O failure while SHA-256 hashing file", path, errno);
        }
        hash.update(std::span<const std::byte>(
            buffer.data(), static_cast<std::size_t>(count)));
    }

    struct stat after{};
    if (::fstat(descriptor.get(), &after) != 0) {
        throw file_hash_system_error(
            "Could not inspect open file after SHA-256 hashing", path, errno);
    }

    const int observed_descriptor_value = ::open(path.c_str(), flags);
    if (observed_descriptor_value < 0) {
        throw file_hash_system_error(
            "SHA-256 input path changed while hashing", path, errno);
    }
    ScopedFileDescriptor observed_descriptor(observed_descriptor_value);
    struct stat observed{};
    if (::fstat(observed_descriptor.get(), &observed) != 0) {
        throw file_hash_system_error(
            "Could not inspect SHA-256 input path after hashing", path, errno);
    }

    if (!same_file_identity(before, after)
        || !same_file_identity(after, observed)) {
        throw std::runtime_error(
            "SHA-256 input changed while hashing: " + path.string());
    }
    return hash.finish_hex();
#else
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error(
            "Could not open file for SHA-256 hashing: " + path.string());
    }

    Sha256Accumulator hash;
    std::array<char, 64U * 1024U> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            hash.update(std::as_bytes(std::span(
                buffer.data(), static_cast<std::size_t>(count))));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "I/O failure while SHA-256 hashing file: " + path.string());
    }
    return hash.finish_hex();
#endif
}

std::string sha256_particle_state(
    std::span<const core::ParticleId> particle_ids,
    std::span<const core::Real> positions_x,
    std::span<const core::Real> positions_y,
    std::span<const core::Real> positions_z) {
    const std::size_t count = particle_ids.size();
    if (count == 0
        || positions_x.size() != count
        || positions_y.size() != count
        || positions_z.size() != count) {
        throw std::invalid_argument(
            "Particle-state SHA-256 requires nonempty equal-sized ID and position arrays");
    }
    static_assert(sizeof(double) == sizeof(std::uint64_t));

    bool strictly_increasing = true;
    for (std::size_t index = 1; index < count; ++index) {
        if (particle_ids[index] <= particle_ids[index - 1]) {
            strictly_increasing = false;
            break;
        }
    }

    std::vector<std::size_t> order;
    if (!strictly_increasing) {
        order.resize(count);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
            if (particle_ids[lhs] != particle_ids[rhs]) {
                return particle_ids[lhs] < particle_ids[rhs];
            }
            return lhs < rhs;
        });
        for (std::size_t index = 1; index < count; ++index) {
            if (particle_ids[order[index]] == particle_ids[order[index - 1]]) {
                throw std::invalid_argument(
                    "Particle-state SHA-256 requires unique particle IDs");
            }
        }
    }

    Sha256Accumulator hash;
    constexpr char domain[] = "cosmo-nbody-particle-position-state";
    hash.update(std::as_bytes(std::span<const char>(domain, sizeof(domain))));

    std::array<std::byte, sizeof(std::uint64_t)> count_bytes{};
    encode_little_endian_u64(static_cast<std::uint64_t>(count), count_bytes.data());
    hash.update(count_bytes);

    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
        const std::size_t index = strictly_increasing ? ordinal : order[ordinal];
        const core::Real coordinates[3] = {
            positions_x[index], positions_y[index], positions_z[index]};
        std::array<std::byte, 4U * sizeof(std::uint64_t)> record{};
        encode_little_endian_u64(particle_ids[index], record.data());
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(coordinates[axis])) {
                throw std::invalid_argument(
                    "Particle-state SHA-256 requires finite particle coordinates");
            }
            const double canonical = coordinates[axis] == core::Real{0.0}
                ? 0.0
                : static_cast<double>(coordinates[axis]);
            std::uint64_t canonical_bits;
            std::memcpy(&canonical_bits, &canonical, sizeof(canonical_bits));
            encode_little_endian_u64(
                canonical_bits,
                record.data() + (axis + 1U) * sizeof(std::uint64_t));
        }
        hash.update(record);
    }
    return hash.finish_hex();
}

void require_file_sha256(
    const std::filesystem::path& path,
    const std::string& expected_sha256,
    const char* context) {
    if (!context) {
        throw std::invalid_argument("SHA-256 verification context must not be null");
    }
    if (!is_canonical_sha256(expected_sha256)) {
        throw std::invalid_argument(
            std::string(context)
            + " expected SHA-256 must contain exactly 64 lowercase hexadecimal characters");
    }
    const std::string observed = sha256_file(path);
    if (observed != expected_sha256) {
        throw std::runtime_error(
            std::string(context) + " content changed after configuration admission: '"
            + path.string() + "'; expected sha256=" + expected_sha256
            + ", observed sha256=" + observed);
    }
}

} // namespace cosmo_nbody::io
