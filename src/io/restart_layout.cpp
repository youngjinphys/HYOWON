#include "cosmo_nbody/io/restart_io.hpp"

#include "cosmo_nbody/io/hdf5_handle.hpp"

#include <hdf5.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace cosmo_nbody::io {
namespace {

std::filesystem::path normalized_layout_path(const std::string& filename) {
    if (filename.empty()) {
        throw std::invalid_argument(
            "Restart layout inspection requires a non-empty path");
    }
    std::filesystem::path path(filename);
    if (path.extension() != ".hdf5") path += ".hdf5";
    return std::filesystem::absolute(path).lexically_normal();
}

void read_layout_attr(
    hid_t location,
    const char* name,
    hid_t type,
    void* value) {
    const std::string context = std::string("restart layout attribute ") + name;
    auto attribute = H5AttributeHandle::checked(
        H5Aopen(location, name, H5P_DEFAULT), context);
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute.get()), context + " dataspace");
    if (H5Sget_simple_extent_type(space.get()) != H5S_SCALAR) {
        throw std::runtime_error(
            std::string("Restart layout attribute must be scalar: ") + name);
    }
    check_hdf5(
        H5Aread(attribute.get(), type, value, context),
        "read " + context);
}

void require_layout_dataset_length(
    hid_t group,
    const char* name,
    std::uint64_t expected) {
    auto dataset = H5DatasetHandle::checked(
        H5Dopen2(group, name, H5P_DEFAULT),
        std::string("restart layout dataset ") + name);
    auto space = H5SpaceHandle::checked(
        H5Dget_space(dataset.get()),
        std::string("restart layout dataspace ") + name);
    if (H5Sget_simple_extent_ndims(space.get()) != 1) {
        throw std::runtime_error(
            std::string("Restart dataset must have rank 1: ") + name);
    }
    hsize_t dimensions[1] = {0};
    check_hdf5(
        H5Sget_simple_extent_dims(space.get(), dimensions, nullptr),
        std::string("read restart layout dimensions ") + name);
    if (dimensions[0] != static_cast<hsize_t>(expected)) {
        throw std::runtime_error(
            std::string("Restart dataset length mismatch: ") + name);
    }
}

} // namespace

RestartReadSession::RestartReadSession(
    ExactHdf5FileIdentity identity) noexcept
    : identity_(std::move(identity)) {}

bool RestartReadSession::matches_path(const std::string& filename) const {
    return identity_.requested_path() == normalized_layout_path(filename);
}

bool RestartReadSession::exact_object_identity_available() const noexcept {
    return identity_.exact_identity_available();
}

std::uint64_t RestartReadSession::exact_object_size_bytes() const {
    return identity_.size_bytes();
}

void RestartReadSession::finish_exact_object_identity() const {
    identity_.finish_identity();
}

std::string RestartReadSession::finish_exact_object_sha256() const {
    return identity_.finish_sha256();
}

RestartReadSession RestartIO::open_restart_session(
    const std::string& filename) const {
    const std::filesystem::path path = normalized_layout_path(filename);
    RestartReadSession session(
        ExactHdf5FileIdentity::open_readonly(
            path, "serial restart exact-object admission"));
    (void)inspect_restart_layout(session);
    return session;
}

RestartLayout RestartIO::inspect_restart_layout(
    const std::string& filename) const {
    auto session = open_restart_session(filename);
    return inspect_restart_layout(session);
}

RestartLayout RestartIO::inspect_restart_layout(
    const RestartReadSession& session) const {
    if (session.file_id() < 0) {
        throw std::invalid_argument(
            "Restart layout inspection requires an open session");
    }
    auto particles = H5GroupHandle::checked(
        H5Gopen2(session.file_id(), "/Particles", H5P_DEFAULT),
        "restart layout Particles group");

    unsigned long long particle_count = 0;
    int uniform_flag = 0;
    double uniform_mass = 0.0;
    read_layout_attr(
        particles.get(), "NumParticles", H5T_NATIVE_ULLONG, &particle_count);
    read_layout_attr(
        particles.get(), "UniformMassFlag", H5T_NATIVE_INT, &uniform_flag);
    read_layout_attr(
        particles.get(), "UniformMass", H5T_NATIVE_DOUBLE, &uniform_mass);

    if (uniform_flag != 0 && uniform_flag != 1) {
        throw std::runtime_error(
            "Restart UniformMassFlag is invalid during layout inspection");
    }
    if (!std::isfinite(uniform_mass) || uniform_mass < 0.0
        || (uniform_flag == 1 && uniform_mass <= 0.0)) {
        throw std::runtime_error(
            "Restart uniform mass is invalid during layout inspection");
    }
    if constexpr (
        sizeof(unsigned long long) > sizeof(std::uint64_t)) {
        if (particle_count > static_cast<unsigned long long>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "Restart particle count exceeds uint64 range");
        }
    }
    const std::uint64_t count = static_cast<std::uint64_t>(particle_count);
    if (count > configured_particle_count()) {
        throw std::runtime_error(
            "Restart particle count exceeds configured N^3 before allocation");
    }

    for (const char* name : {
            "pos_x", "pos_y", "pos_z",
            "mom_x", "mom_y", "mom_z",
            "ParticleIDs"}) {
        require_layout_dataset_length(particles.get(), name, count);
    }
    if (uniform_flag == 0) {
        require_layout_dataset_length(particles.get(), "masses", count);
    }

    return RestartLayout{count, uniform_flag == 1};
}

} // namespace cosmo_nbody::io
