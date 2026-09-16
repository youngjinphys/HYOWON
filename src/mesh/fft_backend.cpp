// FFTW3 backend implementation with optional threaded plan execution.
#include "cosmo_nbody/mesh/fft_backend.hpp"
#include "cosmo_nbody/mesh/fftw_runtime.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/durable_text_publication.hpp"
#include "cosmo_nbody/io/durable_file_publication.hpp"
#include "cosmo_nbody/runtime/raw_scratch_buffer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <thread>
#include <utility>
#include <vector>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody::mesh {
namespace {

constexpr std::size_t MAXIMUM_FFTW_WISDOM_BYTES = 16U * 1024U * 1024U;
constexpr std::size_t MAXIMUM_FFTW_PLAN_TEXT_BYTES = 16U * 1024U * 1024U;
constexpr std::string_view SERIAL_PLANNER_SCHEMA =
    "serial_fftw_r2c_c2r_3d_out_of_place_aligned_and_unaligned";
constexpr std::string_view SERIAL_PLANNER_FLAGS =
    "aligned=FFTW_MEASURE;unaligned=FFTW_MEASURE|FFTW_UNALIGNED";

std::vector<FftwPlanningRecord>& mutable_fftw_planning_records() noexcept {
    static std::vector<FftwPlanningRecord> records;
    return records;
}

std::atomic_flag& fftw_planning_records_lock() noexcept {
    static std::atomic_flag lock = ATOMIC_FLAG_INIT;
    return lock;
}

class FftwPlanningRecordsGuard {
public:
    FftwPlanningRecordsGuard() noexcept {
        auto& lock = fftw_planning_records_lock();
        while (lock.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    ~FftwPlanningRecordsGuard() noexcept {
        fftw_planning_records_lock().clear(std::memory_order_release);
    }

    FftwPlanningRecordsGuard(const FftwPlanningRecordsGuard&) = delete;
    FftwPlanningRecordsGuard& operator=(
        const FftwPlanningRecordsGuard&) = delete;
};

std::once_flag& fftw_policy_report_once_flag() {
    static auto* flag = new std::once_flag();
    return *flag;
}

struct FftwAllocationDeleter {
    template <typename T>
    void operator()(T* pointer) const noexcept {
        fftw_free(pointer);
    }
};

void report_fftw_policy_once() {
    std::call_once(fftw_policy_report_once_flag(), [] {
        const int host_threads = requested_fftw_host_threads();
#ifdef COSMO_NBODY_HAS_FFTW_THREADS
        std::clog << "[fftw] Threaded FFTW plans enabled; new 3D plans use "
                  << host_threads << " thread"
                  << (host_threads == 1 ? "" : "s") << ".\n";
#else
        if (host_threads > 1) {
            std::clog
                << "[fftw] WARNING: host policy requests " << host_threads
                << " threads, but this executable lacks FFTW threaded-plan support; "
                   "3D FFT execution remains serial.\n";
        } else {
            std::clog << "[fftw] Serial FFTW plans enabled.\n";
        }
#endif
    });
}

std::runtime_error wisdom_input_error(
    std::string_view operation,
    const std::filesystem::path& path,
    std::string_view detail = {}) {
    std::string message = "Cannot " + std::string(operation)
        + " FFTW wisdom input " + path.string();
    if (!detail.empty()) message += ": " + std::string(detail);
    return std::runtime_error(std::move(message));
}

#ifdef _WIN32
bool same_file_observation(
    const BY_HANDLE_FILE_INFORMATION& lhs,
    const BY_HANDLE_FILE_INFORMATION& rhs) noexcept {
    return lhs.dwVolumeSerialNumber == rhs.dwVolumeSerialNumber
        && lhs.nFileIndexHigh == rhs.nFileIndexHigh
        && lhs.nFileIndexLow == rhs.nFileIndexLow
        && lhs.nFileSizeHigh == rhs.nFileSizeHigh
        && lhs.nFileSizeLow == rhs.nFileSizeLow
        && lhs.ftLastWriteTime.dwHighDateTime
            == rhs.ftLastWriteTime.dwHighDateTime
        && lhs.ftLastWriteTime.dwLowDateTime
            == rhs.ftLastWriteTime.dwLowDateTime;
}
#else
bool same_file_observation(
    const struct stat& lhs,
    const struct stat& rhs) noexcept {
    if (lhs.st_dev != rhs.st_dev || lhs.st_ino != rhs.st_ino
        || lhs.st_mode != rhs.st_mode || lhs.st_size != rhs.st_size) {
        return false;
    }
#if defined(__APPLE__)
    return lhs.st_mtimespec.tv_sec == rhs.st_mtimespec.tv_sec
        && lhs.st_mtimespec.tv_nsec == rhs.st_mtimespec.tv_nsec
        && lhs.st_ctimespec.tv_sec == rhs.st_ctimespec.tv_sec
        && lhs.st_ctimespec.tv_nsec == rhs.st_ctimespec.tv_nsec;
#else
    return lhs.st_mtim.tv_sec == rhs.st_mtim.tv_sec
        && lhs.st_mtim.tv_nsec == rhs.st_mtim.tv_nsec
        && lhs.st_ctim.tv_sec == rhs.st_ctim.tv_sec
        && lhs.st_ctim.tv_nsec == rhs.st_ctim.tv_nsec;
#endif
}

void close_descriptor_noexcept(int descriptor) noexcept {
    if (descriptor < 0) return;
#if defined(__linux__)
    // Linux releases the numeric descriptor before reporting most close
    // errors, including EINTR, so a retry can close an unrelated reused FD.
    (void)::close(descriptor);
#else
    // Darwin's ordinary close() can be interrupted before descriptor release.
    int status = 0;
    do {
        status = ::close(descriptor);
    } while (status != 0 && errno == EINTR);
#endif
}
#endif

std::optional<std::string> capture_wisdom_file_bytes(
    const std::filesystem::path& path) {
#ifdef _WIN32
    HANDLE handle = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
        nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) {
            return std::nullopt;
        }
        throw wisdom_input_error(
            "open", path, "Windows error " + std::to_string(error));
    }
    BY_HANDLE_FILE_INFORMATION before{};
    FILE_ATTRIBUTE_TAG_INFO attributes{};
    const bool observed = GetFileInformationByHandle(handle, &before) != 0
        && GetFileInformationByHandleEx(
            handle, FileAttributeTagInfo, &attributes, sizeof(attributes)) != 0;
    if (!observed || (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)
        || (attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        CloseHandle(handle);
        throw wisdom_input_error(
            "admit regular non-reparse", path,
            observed ? "object type rejected" : "metadata query failed");
    }
    const std::uint64_t size =
        (static_cast<std::uint64_t>(before.nFileSizeHigh) << 32U)
        | before.nFileSizeLow;
    if (size > MAXIMUM_FFTW_WISDOM_BYTES
        || size > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        CloseHandle(handle);
        throw wisdom_input_error("read bounded", path, "file is too large");
    }
    std::string bytes(static_cast<std::size_t>(size), '\0');
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const DWORD request = static_cast<DWORD>(std::min<std::size_t>(
            bytes.size() - offset,
            static_cast<std::size_t>(std::numeric_limits<DWORD>::max())));
        DWORD received = 0;
        if (!ReadFile(handle, bytes.data() + offset, request, &received, nullptr)
            || received == 0) {
            CloseHandle(handle);
            throw wisdom_input_error("read", path, "Windows ReadFile failed");
        }
        offset += received;
    }
    BY_HANDLE_FILE_INFORMATION after{};
    const bool stable = GetFileInformationByHandle(handle, &after) != 0
        && same_file_observation(before, after);
    CloseHandle(handle);
    if (!stable) {
        throw wisdom_input_error(
            "capture stable", path, "object changed while being read");
    }
#else
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
#ifdef O_NONBLOCK
    flags |= O_NONBLOCK;
#endif
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0) {
        if (errno == ENOENT) return std::nullopt;
        throw wisdom_input_error("open", path, std::strerror(errno));
    }
    struct stat before{};
    if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode)) {
        const int error = errno;
        close_descriptor_noexcept(descriptor);
        throw wisdom_input_error(
            "admit regular non-symlink", path,
            error == 0 ? "object type rejected" : std::strerror(error));
    }
    if (before.st_size < 0
        || static_cast<std::uintmax_t>(before.st_size)
            > MAXIMUM_FFTW_WISDOM_BYTES) {
        close_descriptor_noexcept(descriptor);
        throw wisdom_input_error("read bounded", path, "file is too large");
    }
    std::string bytes(static_cast<std::size_t>(before.st_size), '\0');
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t received = ::read(
            descriptor, bytes.data() + offset, bytes.size() - offset);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) {
            const int error = received < 0 ? errno : 0;
            close_descriptor_noexcept(descriptor);
            throw wisdom_input_error(
                "read exact", path,
                error == 0 ? "unexpected end of file" : std::strerror(error));
        }
        offset += static_cast<std::size_t>(received);
    }
    struct stat after{};
    struct stat named_after{};
    const bool stable = ::fstat(descriptor, &after) == 0
        && same_file_observation(before, after)
        && ::lstat(path.c_str(), &named_after) == 0
        && S_ISREG(named_after.st_mode)
        && named_after.st_dev == after.st_dev
        && named_after.st_ino == after.st_ino;
    close_descriptor_noexcept(descriptor);
    if (!stable) {
        throw wisdom_input_error(
            "capture stable", path, "object changed while being read");
    }
#endif
    if (bytes.find('\0') != std::string::npos) {
        throw wisdom_input_error("admit", path, "embedded NUL byte");
    }
    if (bytes.empty()) {
        throw wisdom_input_error("admit", path, "empty wisdom file");
    }
    return bytes;
}

std::optional<std::filesystem::path> explicit_wisdom_directory() {
    static constexpr const char* environment_name =
        "HYOWON_FFTW_WISDOM_DIRECTORY";
    const char* configured = std::getenv(environment_name);
    if (configured == nullptr) return std::nullopt;
    if (*configured == '\0') {
        throw std::invalid_argument(
            std::string(environment_name)
            + " must be unset or name an existing absolute directory");
    }

    const std::filesystem::path requested(configured);
    if (!requested.is_absolute()) {
        throw std::invalid_argument(
            std::string(environment_name)
            + " must name an existing absolute directory: "
            + requested.string());
    }
    std::error_code error;
    const auto status = std::filesystem::symlink_status(requested, error);
    if (error || std::filesystem::is_symlink(status)
        || !std::filesystem::is_directory(status)) {
        throw std::runtime_error(
            std::string(environment_name)
            + " must name an existing non-symlink directory: "
            + requested.string()
            + (error ? " (" + error.message() + ")" : ""));
    }
    const auto canonical = std::filesystem::canonical(requested, error);
    if (error) {
        throw std::runtime_error(
            "Cannot canonicalize explicit FFTW wisdom directory "
            + requested.string() + ": " + error.message());
    }
    return canonical;
}

void append_identity_field(std::string& target, std::string_view value) {
    target.append(std::to_string(value.size()));
    target.push_back(':');
    target.append(value);
    target.push_back('\0');
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
std::string hexadecimal_u32(std::uint32_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(8) << value;
    return output.str();
}
#endif

#if (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))) \
    || (defined(__linux__) && defined(__aarch64__))
std::string hexadecimal_u64(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}
#endif

#if defined(__APPLE__)
std::string optional_sysctl_text(const char* name) {
    std::size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0
        || size > 64U * 1024U) {
        return {};
    }
    std::string value(size, '\0');
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0
        || size == 0 || size > value.size()) {
        return {};
    }
    value.resize(size);
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}
#endif

std::string runtime_system_identity_material() {
    std::string identity;
#if defined(_WIN32)
    append_identity_field(identity, "windows");
#elif defined(__APPLE__)
    append_identity_field(identity, "macos");
#elif defined(__linux__)
    append_identity_field(identity, "linux");
#else
    append_identity_field(identity, "unknown_os");
#endif

#if defined(__x86_64__) || defined(_M_X64)
    append_identity_field(identity, "x86_64");
#elif defined(__i386__) || defined(_M_IX86)
    append_identity_field(identity, "x86_32");
#elif defined(__aarch64__) || defined(_M_ARM64)
    append_identity_field(identity, "aarch64");
#else
    append_identity_field(identity, "unknown_architecture");
#endif

    append_identity_field(
        identity,
        std::endian::native == std::endian::little
            ? "little_endian"
            : (std::endian::native == std::endian::big
                ? "big_endian"
                : "mixed_endian"));
    append_identity_field(identity, "real_bytes=" + std::to_string(sizeof(core::Real)));
    append_identity_field(identity, "pointer_bytes=" + std::to_string(sizeof(void*)));
    append_identity_field(identity, "size_t_bytes=" + std::to_string(sizeof(std::size_t)));
#ifdef __VERSION__
    append_identity_field(identity, std::string("app_compiler=") + __VERSION__);
#else
    append_identity_field(identity, "app_compiler=unavailable");
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    const unsigned int maximum_basic = __get_cpuid_max(0U, nullptr);
    const unsigned int maximum_extended = __get_cpuid_max(0x80000000U, nullptr);
    append_identity_field(
        identity, "cpuid_max_basic=" + hexadecimal_u32(maximum_basic));
    append_identity_field(
        identity, "cpuid_max_extended=" + hexadecimal_u32(maximum_extended));
    const auto append_cpuid = [&](unsigned int leaf, unsigned int subleaf) {
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
        __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
        append_identity_field(
            identity,
            "cpuid_" + hexadecimal_u32(leaf) + "_"
                + hexadecimal_u32(subleaf) + "="
                + hexadecimal_u32(eax) + hexadecimal_u32(ebx)
                + hexadecimal_u32(ecx) + hexadecimal_u32(edx));
    };
    append_cpuid(0U, 0U);
    if (maximum_basic >= 1U) {
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
        __cpuid_count(1U, 0U, eax, ebx, ecx, edx);
        // EBX[31:24] is the initial APIC ID of the logical processor on which
        // this query ran. It is placement identity, not an FFTW capability;
        // retaining it would fragment one host's cache across cores and make
        // simultaneous identical publishers miss each other entirely.
        ebx &= 0x00ffffffU;
        append_identity_field(
            identity,
            "cpuid_00000001_00000000="
                + hexadecimal_u32(eax) + hexadecimal_u32(ebx)
                + hexadecimal_u32(ecx) + hexadecimal_u32(edx));
        constexpr unsigned int OSXSAVE = 1U << 27U;
        if ((ecx & OSXSAVE) != 0U) {
            std::uint32_t xcr0_low = 0;
            std::uint32_t xcr0_high = 0;
            __asm__ volatile(
                "xgetbv"
                : "=a"(xcr0_low), "=d"(xcr0_high)
                : "c"(0));
            append_identity_field(
                identity,
                "xcr0=" + hexadecimal_u64(
                    (static_cast<std::uint64_t>(xcr0_high) << 32U)
                    | xcr0_low));
        }
    }
    if (maximum_basic >= 7U) append_cpuid(7U, 0U);
    if (maximum_extended >= 0x80000001U) {
        append_cpuid(0x80000001U, 0U);
    }
#elif defined(__linux__) && defined(__aarch64__)
    append_identity_field(
        identity,
        "auxv_hwcap=" + hexadecimal_u64(
            static_cast<std::uint64_t>(getauxval(AT_HWCAP))));
#ifdef AT_HWCAP2
    append_identity_field(
        identity,
        "auxv_hwcap2=" + hexadecimal_u64(
            static_cast<std::uint64_t>(getauxval(AT_HWCAP2))));
#endif
#elif defined(__APPLE__)
    append_identity_field(
        identity, "hw_machine=" + optional_sysctl_text("hw.machine"));
    append_identity_field(
        identity, "hw_model=" + optional_sysctl_text("hw.model"));
    append_identity_field(
        identity,
        "cpu_brand=" + optional_sysctl_text("machdep.cpu.brand_string"));
#endif
    return identity;
}

std::filesystem::path loaded_image_path_for_address(const void* address) {
    if (address == nullptr) {
        throw std::invalid_argument(
            "Loaded FFTW image identity requires a non-null symbol address");
    }
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(address),
            &module)) {
        throw std::runtime_error(
            "Could not resolve the loaded FFTW provider module");
    }
    std::vector<wchar_t> buffer(1024U, L'\0');
    while (true) {
        const DWORD length = GetModuleFileNameW(
            module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            throw std::runtime_error(
                "Could not resolve the loaded FFTW provider path");
        }
        if (length < buffer.size()) {
            return std::filesystem::canonical(std::filesystem::path(
                std::wstring(buffer.data(), length)));
        }
        if (buffer.size() > std::numeric_limits<DWORD>::max() / 2U) {
            throw std::runtime_error(
                "Loaded FFTW provider path exceeds DWORD capacity");
        }
        buffer.resize(buffer.size() * 2U, L'\0');
    }
#else
    Dl_info information{};
    if (dladdr(address, &information) == 0
        || information.dli_fname == nullptr
        || *information.dli_fname == '\0') {
        throw std::runtime_error(
            "Could not resolve the loaded FFTW provider image");
    }
    return std::filesystem::canonical(
        std::filesystem::path(information.dli_fname));
#endif
}

const void* loaded_symbol_address(
    const char* name,
    const void* fallback) {
#ifdef _WIN32
    (void)name;
    return fallback;
#else
    if (void* resolved = dlsym(RTLD_DEFAULT, name); resolved != nullptr) {
        return resolved;
    }
    return fallback;
#endif
}

std::string loaded_fftw_provider_identity() {
    std::vector<std::pair<std::string, std::string>> providers;
    const auto add_provider = [&](const char* label, const void* address) {
        const auto path = loaded_image_path_for_address(address);
        providers.emplace_back(label, io::sha256_file(path));
    };
    add_provider(
        "base",
        loaded_symbol_address(
            "fftw_plan_dft_r2c_3d", static_cast<const void*>(fftw_version)));
#ifdef COSMO_NBODY_HAS_FFTW_THREADS
    add_provider(
        "threads",
        loaded_symbol_address(
            "fftw_init_threads", static_cast<const void*>(fftw_version)));
#endif
#ifdef COSMO_NBODY_HAS_FFTW_MPI
    if (const void* mpi_provider = loaded_symbol_address(
            "fftw_mpi_plan_dft_r2c_3d", nullptr);
        mpi_provider != nullptr) {
        add_provider("mpi", mpi_provider);
    }
#endif
    std::sort(providers.begin(), providers.end());
    std::string identity;
    for (const auto& [label, digest] : providers) {
        append_identity_field(identity, label);
        append_identity_field(identity, digest);
    }
    return identity;
}

std::string printed_plan_text(fftw_plan plan, std::string_view label) {
    if (plan == nullptr) {
        throw std::invalid_argument(
            "Cannot identify a null FFTW plan: " + std::string(label));
    }
    char* printed = fftw_sprint_plan(plan);
    if (printed == nullptr) {
        throw std::runtime_error(
            "FFTW could not print plan representation: "
            + std::string(label));
    }
    std::string text;
    try {
        const std::size_t length = std::strlen(printed);
        if (length > MAXIMUM_FFTW_PLAN_TEXT_BYTES) {
            throw std::length_error(
                "FFTW printed plan representation exceeds its bound");
        }
        text.assign(printed, length);
    } catch (...) {
        fftw_free(printed);
        throw;
    }
    fftw_free(printed);
    return text;
}

std::string serial_plan_representation_sha256(
    fftw_plan forward,
    fftw_plan inverse,
    fftw_plan forward_unaligned,
    fftw_plan inverse_unaligned,
    int real_alignment,
    int complex_alignment) {
    std::string material;
    append_identity_field(material, SERIAL_PLANNER_SCHEMA);
    append_identity_field(material, SERIAL_PLANNER_FLAGS);
    append_identity_field(
        material, "real_alignment=" + std::to_string(real_alignment));
    append_identity_field(
        material, "complex_alignment=" + std::to_string(complex_alignment));
    const auto append_plan = [&](std::string_view label, fftw_plan plan) {
        append_identity_field(material, label);
        append_identity_field(material, printed_plan_text(plan, label));
    };
    append_plan("forward_aligned", forward);
    append_plan("inverse_aligned", inverse);
    append_plan("forward_unaligned", forward_unaligned);
    append_plan("inverse_unaligned", inverse_unaligned);
    return io::sha256_text(material);
}

std::string current_planner_wisdom_bytes() {
    char* exported = fftw_export_wisdom_to_string();
    if (exported == nullptr) {
        throw std::runtime_error(
            "FFTW could not export planner wisdom for numerical provenance");
    }
    std::string bytes;
    try {
        bytes.assign(exported);
    } catch (...) {
        fftw_free(exported);
        throw;
    }
    fftw_free(exported);
    if (bytes.size() > MAXIMUM_FFTW_WISDOM_BYTES) {
        throw std::length_error(
            "FFTW exported wisdom exceeds its persisted byte bound");
    }
    if (bytes.find('\0') != std::string::npos) {
        throw std::runtime_error(
            "FFTW exported wisdom contains an embedded NUL byte");
    }
    return bytes;
}

} // namespace

std::string fftw_build_identity_sha256() {
    std::string identity(fftw_version);
    identity.push_back('\0');
    identity.append(fftw_cc);
    identity.push_back('\0');
    identity.append(fftw_codelet_optim);
    return io::sha256_text(identity);
}

std::string fftw_provider_path_content_sha256() {
    static const std::string digest = io::sha256_text(
        loaded_fftw_provider_identity());
    return digest;
}

std::string fftw_runtime_system_identity_sha256() {
    static const std::string digest = io::sha256_text(
        runtime_system_identity_material());
    return digest;
}

void append_fftw_planning_record(FftwPlanningRecord record) {
    FftwPlanningRecordsGuard lock;
    mutable_fftw_planning_records().push_back(std::move(record));
}

void reserve_fftw_planning_record_slot() {
    FftwPlanningRecordsGuard lock;
    auto& records = mutable_fftw_planning_records();
    if (records.size() == std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("FFTW planning-record registry is full");
    }
    records.reserve(records.size() + 1U);
}

void append_reserved_fftw_planning_record(
    FftwPlanningRecord record) noexcept {
    static_assert(std::is_nothrow_move_constructible_v<FftwPlanningRecord>);
    FftwPlanningRecordsGuard lock;
    auto& records = mutable_fftw_planning_records();
    if (records.size() == records.capacity()) std::terminate();
    records.push_back(std::move(record));
}

std::vector<FftwPlanningRecord> fftw_planning_records_snapshot() {
    FftwPlanningRecordsGuard lock;
    return mutable_fftw_planning_records();
}

FFTBackend::FFTBackend(
    const MeshGeometry& geom,
    config::ScratchMode planning_scratch_mode,
    const std::string& scratch_directory)
    : geom_(geom),
      N_(geom.grid_size()),
      real_size_(geom.real_size()),
      complex_size_(geom.complex_size()),
      norm_factor_(1.0 / (static_cast<core::Real>(geom.grid_size())
                          * static_cast<core::Real>(geom.grid_size())
                          * static_cast<core::Real>(geom.grid_size()))),
      forward_plan_(nullptr),
      inverse_plan_(nullptr),
      forward_unaligned_plan_(nullptr),
      inverse_unaligned_plan_(nullptr) {
    if (geom_.local_n0() != N_ || geom_.local_0_start() != 0) {
        throw std::invalid_argument(
            "FFTBackend currently requires single-rank full-mesh geometry");
    }
    if (N_ > static_cast<std::size_t>(INT_MAX)) {
        throw std::overflow_error("FFTW 3D plan dimension exceeds int range");
    }

    std::lock_guard<std::mutex> planner_lock(shared_fftw_planner_mutex());
    fftw_forget_wisdom();

    int fftw_thread_count = 1;
#ifdef COSMO_NBODY_HAS_FFTW_THREADS
    require_fftw_threads_initialized();
    fftw_thread_count = requested_fftw_host_threads();
    configure_fftw_plan_threads(fftw_thread_count);
#endif
    report_fftw_policy_once();

    if (real_size_ > std::numeric_limits<std::size_t>::max() / sizeof(core::Real)
        || complex_size_ > std::numeric_limits<std::size_t>::max() / sizeof(fftw_complex)) {
        throw std::overflow_error("FFTW planner array byte size overflows size_t");
    }
    const std::size_t real_bytes = real_size_ * sizeof(core::Real);
    const std::size_t complex_bytes = complex_size_ * sizeof(fftw_complex);
    std::unique_ptr<runtime::RawScratchBuffer> real_backing;
    std::unique_ptr<runtime::RawScratchBuffer> complex_backing;
    std::unique_ptr<core::Real, FftwAllocationDeleter> anonymous_real;
    std::unique_ptr<fftw_complex, FftwAllocationDeleter> anonymous_complex;
    core::Real* dummy_real = nullptr;
    fftw_complex* dummy_complex = nullptr;
    if (config::uses_file_backed_scratch(planning_scratch_mode)) {
        real_backing = std::make_unique<runtime::RawScratchBuffer>(
            real_bytes, planning_scratch_mode, scratch_directory, "fftw_plan_real");
        complex_backing = std::make_unique<runtime::RawScratchBuffer>(
            complex_bytes, planning_scratch_mode, scratch_directory, "fftw_plan_complex");
        dummy_real = static_cast<core::Real*>(real_backing->data());
        dummy_complex = static_cast<fftw_complex*>(complex_backing->data());
        // Start trivial scalar lifetimes without touching every mapped page.
        std::uninitialized_default_construct_n(dummy_real, real_size_);
        // Placement array new starts double[2] lifetimes on older libc++.
        ::new (static_cast<void*>(dummy_complex)) fftw_complex[complex_size_];
    } else {
        anonymous_real.reset(fftw_alloc_real(real_size_));
        anonymous_complex.reset(fftw_alloc_complex(complex_size_));
        dummy_real = anonymous_real.get();
        dummy_complex = anonymous_complex.get();
    }
    if (!dummy_real || !dummy_complex) throw std::bad_alloc();
    real_alignment_ = fftw_alignment_of(dummy_real);
    complex_alignment_ = fftw_alignment_of(
        reinterpret_cast<core::Real*>(dummy_complex));
    std::clog << "[fftw] planning_scratch_mode="
              << config::scratch_mode_name(planning_scratch_mode)
              << " real_bytes=" << real_bytes
              << " complex_bytes=" << complex_bytes
              << " real_alignment_class=" << real_alignment_
              << " complex_alignment_class=" << complex_alignment_ << '\n';

    FftwPlanningRecord planning_record;
    planning_record.backend = "serial_fftw";
    planning_record.planner_rigor = "measure";
    planning_record.planner_schema = std::string(SERIAL_PLANNER_SCHEMA);
    planning_record.planner_flags = std::string(SERIAL_PLANNER_FLAGS);
    planning_record.plan_identity_scope =
        "serial_printed_plan_hash_and_post_planning_wisdom_corpus";
    planning_record.grid_size = N_;
    planning_record.thread_count = fftw_thread_count;
    planning_record.mpi_rank_count = 1;
    planning_record.fftw_version = std::string(fftw_version);
    planning_record.fftw_build_identity_sha256 =
        fftw_build_identity_sha256();
    planning_record.fftw_provider_path_content_sha256 =
        fftw_provider_path_content_sha256();
    planning_record.runtime_system_identity_sha256 =
        fftw_runtime_system_identity_sha256();

    const auto wisdom_directory = explicit_wisdom_directory();
    std::filesystem::path wisdom_path;
    bool wisdom_file_existed = false;
    if (wisdom_directory.has_value()) {
        planning_record.wisdom_cache_policy =
            "explicit_cooperative_create_once";
        std::string cache_identity;
        append_identity_field(
            cache_identity,
            planning_record.fftw_build_identity_sha256);
        append_identity_field(
            cache_identity,
            planning_record.fftw_provider_path_content_sha256);
        append_identity_field(
            cache_identity,
            planning_record.runtime_system_identity_sha256);
        append_identity_field(cache_identity, planning_record.planner_schema);
        append_identity_field(cache_identity, planning_record.planner_flags);
        append_identity_field(
            cache_identity,
            "real_alignment=" + std::to_string(real_alignment_));
        append_identity_field(
            cache_identity,
            "complex_alignment=" + std::to_string(complex_alignment_));
        const std::string cache_identity_sha256 =
            io::sha256_text(cache_identity);
        wisdom_path = *wisdom_directory
            / ("fftw_wisdom_"
               + cache_identity_sha256 + "_"
               + std::to_string(N_) + "_"
               + std::to_string(fftw_thread_count) + ".bin");
        planning_record.wisdom_path = wisdom_path.string();
        if (const auto captured = capture_wisdom_file_bytes(wisdom_path);
            captured.has_value()) {
            wisdom_file_existed = true;
            planning_record.imported_wisdom_sha256 =
                io::sha256_text(*captured);
            planning_record.canonical_wisdom_sha256 =
                planning_record.imported_wisdom_sha256;
            if (!fftw_import_wisdom_from_string(captured->c_str())) {
                fftw_forget_wisdom();
                throw std::runtime_error(
                    "Explicit FFTW wisdom was rejected by the linked planner: "
                    + wisdom_path.string());
            }
            planning_record.wisdom_imported = true;
            planning_record.wisdom_publication_outcome =
                "imported_existing";
            std::clog << "[fftw] Imported hash-bound explicit wisdom from "
                      << wisdom_path.string() << '\n';
        } else {
            planning_record.wisdom_publication_outcome =
                "pending_create_once";
        }
    } else {
        planning_record.wisdom_cache_policy = "disabled_and_cleared";
        planning_record.wisdom_publication_outcome = "not_requested";
        std::clog
            << "[fftw] Wisdom cache disabled; planning from an empty state.\n";
    }

    const int n = static_cast<int>(N_);
    forward_plan_ = fftw_plan_dft_r2c_3d(
        n, n, n, dummy_real, dummy_complex, FFTW_MEASURE);
    if (!forward_plan_) {
        throw std::runtime_error("Failed to create FFTW forward plan");
    }

    inverse_plan_ = fftw_plan_dft_c2r_3d(
        n, n, n, dummy_complex, dummy_real, FFTW_MEASURE);
    if (!inverse_plan_) {
        fftw_destroy_plan(forward_plan_);
        forward_plan_ = nullptr;
        throw std::runtime_error("Failed to create FFTW inverse plan");
    }

    forward_unaligned_plan_ = fftw_plan_dft_r2c_3d(
        n,
        n,
        n,
        dummy_real,
        dummy_complex,
        FFTW_MEASURE | FFTW_UNALIGNED);
    if (!forward_unaligned_plan_) {
        fftw_destroy_plan(inverse_plan_);
        inverse_plan_ = nullptr;
        fftw_destroy_plan(forward_plan_);
        forward_plan_ = nullptr;
        throw std::runtime_error(
            "Failed to create FFTW unaligned forward plan");
    }

    inverse_unaligned_plan_ = fftw_plan_dft_c2r_3d(
        n,
        n,
        n,
        dummy_complex,
        dummy_real,
        FFTW_MEASURE | FFTW_UNALIGNED);
    if (!inverse_unaligned_plan_) {
        fftw_destroy_plan(forward_unaligned_plan_);
        forward_unaligned_plan_ = nullptr;
        fftw_destroy_plan(inverse_plan_);
        inverse_plan_ = nullptr;
        fftw_destroy_plan(forward_plan_);
        forward_plan_ = nullptr;
        throw std::runtime_error(
            "Failed to create FFTW unaligned inverse plan");
    }

    try {
        planning_record.plan_representation_sha256 =
            serial_plan_representation_sha256(
                forward_plan_,
                inverse_plan_,
                forward_unaligned_plan_,
                inverse_unaligned_plan_,
                real_alignment_,
                complex_alignment_);
        const std::string planned_wisdom = current_planner_wisdom_bytes();
        planning_record.planned_wisdom_corpus_sha256 =
            io::sha256_text(planned_wisdom);
        if (!wisdom_path.empty() && !wisdom_file_existed) {
            const std::string expected_sha256 =
                planning_record.planned_wisdom_corpus_sha256;
            try {
                io::write_text_durable_atomic_validated(
                    wisdom_path,
                    planned_wisdom,
                    "FFTW wisdom",
                    [expected_sha256](
                        const std::filesystem::path& staged_path) {
                        io::require_file_sha256(
                            staged_path,
                            expected_sha256,
                            "FFTW wisdom staged exact-byte validation");
                    });
                planning_record.wisdom_published = true;
                planning_record.canonical_wisdom_sha256 = expected_sha256;
                planning_record.wisdom_publication_outcome = "created";
                std::clog << "[fftw] Published create-once wisdom to "
                          << wisdom_path.string() << '\n';
            } catch (const io::DurableFilePublicationCollision&) {
                const auto winner_bytes =
                    capture_wisdom_file_bytes(wisdom_path);
                if (!winner_bytes.has_value()
                    || !fftw_import_wisdom_from_string(
                        winner_bytes->c_str())) {
                    throw std::runtime_error(
                        "FFTW wisdom create-only collision exposed an incompatible destination: "
                        + wisdom_path.string());
                }
                planning_record.wisdom_publication_outcome =
                    "compatible_existing_won_race";
                planning_record.canonical_wisdom_sha256 =
                    io::sha256_text(*winner_bytes);
                std::clog
                    << "[fftw] Another process published compatible create-once wisdom first at "
                    << wisdom_path.string() << '\n';
            }
        }
        append_fftw_planning_record(std::move(planning_record));
    } catch (...) {
        fftw_destroy_plan(inverse_unaligned_plan_);
        inverse_unaligned_plan_ = nullptr;
        fftw_destroy_plan(forward_unaligned_plan_);
        forward_unaligned_plan_ = nullptr;
        fftw_destroy_plan(inverse_plan_);
        inverse_plan_ = nullptr;
        fftw_destroy_plan(forward_plan_);
        forward_plan_ = nullptr;
        throw;
    }
}

FFTBackend::~FFTBackend() {
    std::lock_guard<std::mutex> planner_lock(shared_fftw_planner_mutex());
    if (forward_plan_) fftw_destroy_plan(forward_plan_);
    if (inverse_plan_) fftw_destroy_plan(inverse_plan_);
    if (forward_unaligned_plan_) fftw_destroy_plan(forward_unaligned_plan_);
    if (inverse_unaligned_plan_) fftw_destroy_plan(inverse_unaligned_plan_);
}

void FFTBackend::forward(RealField& in, ComplexField& out) {
    if (in.size() != real_size_ || out.size() != complex_size_) {
        throw std::invalid_argument("FFTBackend::forward: invalid field size");
    }
    const bool aligned =
        fftw_alignment_of(in.data()) == real_alignment_
        && fftw_alignment_of(reinterpret_cast<core::Real*>(out.data()))
            == complex_alignment_;
    fftw_execute_dft_r2c(
        aligned ? forward_plan_ : forward_unaligned_plan_,
        in.data(), reinterpret_cast<fftw_complex*>(out.data()));
}

void FFTBackend::forward_external_output(
    RealField& in,
    std::complex<core::Real>* out,
    std::size_t out_size) {
    if (in.size() != real_size_ || !out || out_size != complex_size_) {
        throw std::invalid_argument(
            "FFTBackend::forward_external_output: invalid field size or null storage");
    }
    const bool aligned =
        fftw_alignment_of(in.data()) == real_alignment_
        && fftw_alignment_of(reinterpret_cast<core::Real*>(out))
            == complex_alignment_;
    fftw_execute_dft_r2c(
        aligned ? forward_plan_ : forward_unaligned_plan_,
        in.data(), reinterpret_cast<fftw_complex*>(out));
}

void FFTBackend::normalize_inverse_buffer(
    core::Real* out,
    std::size_t out_size) const {
    if (!out || out_size != real_size_) {
        throw std::invalid_argument(
            "FFTBackend inverse normalization: invalid external buffer");
    }
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(out_size >= 32768)
#endif
    for (std::size_t index = 0; index < out_size; ++index) {
        out[index] *= norm_factor_;
    }
}

void FFTBackend::inverse(ComplexField& in_k, RealField& out_r) {
    if (in_k.size() != complex_size_ || out_r.size() != real_size_) {
        throw std::invalid_argument("FFTBackend::inverse: invalid field size");
    }
    const bool aligned =
        fftw_alignment_of(reinterpret_cast<core::Real*>(in_k.data()))
            == complex_alignment_
        && fftw_alignment_of(out_r.data()) == real_alignment_;
    fftw_execute_dft_c2r(
        aligned ? inverse_plan_ : inverse_unaligned_plan_,
        reinterpret_cast<fftw_complex*>(in_k.data()), out_r.data());
    normalize_inverse_buffer(out_r.data(), out_r.size());
}

void FFTBackend::forward_external(
    core::Real* in,
    std::size_t in_size,
    ComplexField& out) {
    if (!in || in_size != real_size_ || out.size() != complex_size_) {
        throw std::invalid_argument(
            "FFTBackend::forward_external: invalid field size or null storage");
    }
    const bool aligned =
        fftw_alignment_of(in) == real_alignment_
        && fftw_alignment_of(reinterpret_cast<core::Real*>(out.data()))
            == complex_alignment_;
    fftw_execute_dft_r2c(
        aligned ? forward_plan_ : forward_unaligned_plan_,
        in, reinterpret_cast<fftw_complex*>(out.data()));
}

void FFTBackend::inverse_external(
    ComplexField& in,
    core::Real* out,
    std::size_t out_size) {
    if (in.size() != complex_size_ || !out || out_size != real_size_) {
        throw std::invalid_argument(
            "FFTBackend::inverse_external: invalid field size or null storage");
    }
    const bool aligned =
        fftw_alignment_of(reinterpret_cast<core::Real*>(in.data()))
            == complex_alignment_
        && fftw_alignment_of(out) == real_alignment_;
    fftw_execute_dft_c2r(
        aligned ? inverse_plan_ : inverse_unaligned_plan_,
        reinterpret_cast<fftw_complex*>(in.data()), out);
    normalize_inverse_buffer(out, out_size);
}

} // namespace cosmo_nbody::mesh
