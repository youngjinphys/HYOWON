#include "cosmo_nbody/io/parallel_snapshot_io.hpp"

#include "cosmo_nbody/cosmology/units.hpp"
#include "cosmo_nbody/domain/mpi_global_id_check.hpp"
#include "cosmo_nbody/io/bounded_hdf5_string.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/durable_file_publication.hpp"
#include "cosmo_nbody/io/hdf5_handle.hpp"
#include "cosmo_nbody/io/metadata.hpp"
#include "cosmo_nbody/io/output_schema.hpp"
#include "cosmo_nbody/io/snapshot_descriptor.hpp"
#include "cosmo_nbody/io/snapshot_io.hpp"
#include "cosmo_nbody/io/snapshot_physics.hpp"
#include "cosmo_nbody/io/verified_snapshot_source.hpp"
#include "cosmo_nbody/math/mpi_exact_uint64_sum.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"
#include "snapshot_hdf5_file_type.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace io {

ParallelSnapshotIO::ParallelSnapshotIO(
    const config::SimulationParameters& config)
    : ParallelSnapshotIO(config, ".") {}

ParallelSnapshotIO::ParallelSnapshotIO(
    const config::SimulationParameters& config,
    std::filesystem::path output_directory)
    : config_(config),
      output_directory_(std::move(output_directory).lexically_normal()) {
    if (output_directory_.empty()) {
        throw std::invalid_argument(
            "Distributed snapshot output directory must not be empty");
    }
}

namespace {

#ifdef COSMO_NBODY_HAS_MPI

void write_attr(
    hid_t location,
    const std::string& name,
    hid_t type,
    const void* data) {
    auto space = H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR),
        "parallel snapshot scalar attribute space " + name);
    auto attr = H5AttributeHandle::checked(
        H5Acreate2(
            location, name.c_str(),
            detail::canonical_snapshot_file_type(
                type, "parallel snapshot attribute " + name),
            space.get(),
            H5P_DEFAULT, H5P_DEFAULT),
        "parallel snapshot attribute " + name);
    check_hdf5(
        H5Awrite(attr.get(), type, data),
        "write parallel snapshot attribute " + name);
}

void write_attr_array(
    hid_t location,
    const std::string& name,
    hid_t type,
    const void* data,
    hsize_t count) {
    const hsize_t dims[1] = {count};
    auto space = H5SpaceHandle::checked(
        H5Screate_simple(1, dims, nullptr),
        "parallel snapshot array attribute space " + name);
    auto attr = H5AttributeHandle::checked(
        H5Acreate2(
            location, name.c_str(),
            detail::canonical_snapshot_file_type(
                type, "parallel snapshot array attribute " + name),
            space.get(),
            H5P_DEFAULT, H5P_DEFAULT),
        "parallel snapshot attribute " + name);
    check_hdf5(
        H5Awrite(attr.get(), type, data),
        "write parallel snapshot attribute " + name);
}

void write_string_attr(
    hid_t location,
    const std::string& name,
    const std::string& value) {
    auto space = H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR),
        "parallel snapshot string attribute space " + name);
    auto type = H5TypeHandle::checked(
        H5Tcopy(H5T_C_S1),
        "parallel snapshot string attribute type " + name);
    check_hdf5(
        H5Tset_cset(type.get(), H5T_CSET_UTF8),
        "set parallel snapshot UTF-8 string character set " + name);
    check_hdf5(
        H5Tset_size(type.get(), value.size() + 1),
        "set parallel snapshot string attribute size " + name);
    check_hdf5(
        H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
        "set parallel snapshot string padding " + name);
    auto attr = H5AttributeHandle::checked(
        H5Acreate2(
            location, name.c_str(), type.get(), space.get(),
            H5P_DEFAULT, H5P_DEFAULT),
        "parallel snapshot string attribute " + name);
    check_hdf5(
        H5Awrite(attr.get(), type.get(), value.c_str()),
        "write parallel snapshot string attribute " + name);
}

void require_mpi_success(int code, const char* context) {
    if (code != MPI_SUCCESS) {
        // RuntimeContext installs MPI_ERRORS_RETURN so validation failures can
        // be synchronized explicitly. A real communicator/transport error is
        // different: ordinary MPI has no ULFM recovery contract, and a local
        // exception can strand a peer in the matching send/receive. Terminate
        // the whole job; the throw is only a fallback if MPI_Abort returns.
        (void)MPI_Abort(MPI_COMM_WORLD, code);
        throw std::runtime_error(std::string(context) + " failed");
    }
}

void synchronize_distributed_snapshot_exception(
    std::exception_ptr local_exception,
    int mpi_size,
    const char* context) {
    if (mpi_size < 1 || context == nullptr || *context == '\0') {
        throw std::invalid_argument(
            "Distributed snapshot exception synchronization is invalid");
    }
    const int local_failed = local_exception ? 1 : 0;
    int any_failed = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_failed, &any_failed, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for distributed snapshot exception synchronization");
    if (any_failed == 0) return;
    if (local_exception) std::rethrow_exception(local_exception);
    throw std::runtime_error(
        std::string(context) + " failed on another MPI rank");
}

std::string snapshot_exception_text(
    const std::exception_ptr& exception) {
    if (!exception) return "none";
    try {
        std::rethrow_exception(exception);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return "unknown non-standard exception";
    }
}

std::exception_ptr combine_snapshot_exceptions(
    const std::exception_ptr& primary,
    const std::exception_ptr& cleanup,
    const char* context) noexcept {
    if (!primary) return cleanup;
    if (!cleanup) return primary;
    try {
        return std::make_exception_ptr(std::runtime_error(
            std::string(context)
            + ": primary failure (" + snapshot_exception_text(primary)
            + "); cleanup failure (" + snapshot_exception_text(cleanup)
            + "). Inspection state was retained."));
    } catch (...) {
        // Cleanup failure is the evidence-retention risk. Preserve it if a
        // diagnostic string cannot be allocated on the failing rank.
        return cleanup;
    }
}

void require_exact_rank_string_agreement(
    const std::string& local_value,
    const char* label) {
    int size = 0;
    int rank = 0;
    require_mpi_success(
        MPI_Comm_size(MPI_COMM_WORLD, &size),
        "MPI_Comm_size for rank string agreement");
    require_mpi_success(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank for rank string agreement");
    if (size < 1 || rank < 0 || rank >= size) {
        throw std::runtime_error("Invalid MPI topology for rank string agreement");
    }

    const int local_oversize =
        local_value.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())
        ? 1 : 0;
    int any_oversize = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_oversize, &any_oversize, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for rank string size");
    if (any_oversize != 0) {
        throw std::overflow_error(
            std::string(label) + " exceeds MPI int count range");
    }

    const int local_size = static_cast<int>(local_value.size());
    int min_size = 0;
    int max_size = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_size, &min_size, 1,
            MPI_INT, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce for minimum rank string size");
    require_mpi_success(
        MPI_Allreduce(
            &local_size, &max_size, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for maximum rank string size");
    if (min_size != max_size) {
        throw std::invalid_argument(
            std::string(label) + " differs across MPI ranks");
    }

    std::vector<char> reference;
    int local_allocation_failed = 0;
    try {
        reference.resize(static_cast<std::size_t>(local_size));
        if (rank == 0 && local_size > 0) {
            std::copy(local_value.begin(), local_value.end(), reference.begin());
        }
    } catch (...) {
        local_allocation_failed = 1;
    }
    int any_allocation_failed = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_allocation_failed,
            &any_allocation_failed,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD),
        "MPI_Allreduce for rank string reference allocation");
    if (any_allocation_failed != 0) {
        throw std::runtime_error(
            std::string(label)
            + " reference allocation failed on at least one MPI rank");
    }
    require_mpi_success(
        MPI_Bcast(
            local_size == 0 ? nullptr : reference.data(),
            local_size, MPI_CHAR, 0, MPI_COMM_WORLD),
        "MPI_Bcast for rank string agreement");

    const int local_mismatch = std::equal(
        local_value.begin(), local_value.end(), reference.begin()) ? 0 : 1;
    int any_mismatch = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_mismatch, &any_mismatch, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for rank string comparison");
    if (any_mismatch != 0) {
        throw std::invalid_argument(
            std::string(label) + " differs across MPI ranks");
    }
}

H5SpaceHandle make_memory_space_1d(std::size_t count) {
    const hsize_t dims[1] = {
        static_cast<hsize_t>(std::max<std::size_t>(1, count))};
    auto space = H5SpaceHandle::checked(
        H5Screate_simple(1, dims, nullptr),
        "parallel snapshot 1D memory space");
    if (count == 0) {
        check_hdf5(
            H5Sselect_none(space.get()),
            "select none in empty 1D memory space");
    }
    return space;
}

H5SpaceHandle make_memory_space_2d(std::size_t count) {
    const hsize_t dims[2] = {
        static_cast<hsize_t>(std::max<std::size_t>(1, count)), 3};
    auto space = H5SpaceHandle::checked(
        H5Screate_simple(2, dims, nullptr),
        "parallel snapshot 2D memory space");
    if (count == 0) {
        check_hdf5(
            H5Sselect_none(space.get()),
            "select none in empty 2D memory space");
    }
    return space;
}

void select_file_hyperslab_1d(
    hid_t file_space,
    std::uint64_t offset,
    std::size_t count) {
    if (count == 0) {
        check_hdf5(
            H5Sselect_none(file_space),
            "select none in empty 1D file hyperslab");
        return;
    }
    const hsize_t start[1] = {static_cast<hsize_t>(offset)};
    const hsize_t extent[1] = {static_cast<hsize_t>(count)};
    check_hdf5(
        H5Sselect_hyperslab(
            file_space, H5S_SELECT_SET, start,
            nullptr, extent, nullptr),
        "select parallel snapshot 1D file hyperslab");
}

void select_file_hyperslab_2d(
    hid_t file_space,
    std::uint64_t offset,
    std::size_t count) {
    if (count == 0) {
        check_hdf5(
            H5Sselect_none(file_space),
            "select none in empty 2D file hyperslab");
        return;
    }
    const hsize_t start[2] = {
        static_cast<hsize_t>(offset), 0};
    const hsize_t extent[2] = {
        static_cast<hsize_t>(count), 3};
    check_hdf5(
        H5Sselect_hyperslab(
            file_space, H5S_SELECT_SET, start,
            nullptr, extent, nullptr),
        "select parallel snapshot 2D file hyperslab");
}

struct BatchBuffers {
    std::vector<core::Real> coordinates;
    std::vector<core::Real> velocities;
    std::vector<core::ParticleId> ids;
    std::vector<core::Real> masses;
};

constexpr std::size_t snapshot_payload_digest_field_count = 5U;
constexpr std::size_t snapshot_payload_identity_bytes =
    snapshot_payload_digest_field_count * SHA256_HEX_CHARACTER_COUNT;
using SnapshotPayloadIdentity =
    std::array<char, snapshot_payload_identity_bytes>;

SnapshotPayloadIdentity compute_expected_snapshot_payload_identity(
    const core::ParticleStore& particles,
    core::Real current_a,
    std::uint64_t transfer_batch,
    bool include_explicit_masses,
    BatchBuffers& scratch) {
    const std::size_t local_count = particles.num_owned_particles();
    const core::Real inv_a = core::Real{1.0} / current_a;
    Sha256Accumulator coordinates;
    Sha256Accumulator raw_velocities;
    Sha256Accumulator momenta;
    Sha256Accumulator ids;
    Sha256Accumulator masses;

    for (std::size_t begin = 0; begin < local_count;) {
        const std::size_t count = std::min(
            static_cast<std::size_t>(transfer_batch),
            local_count - begin);
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t index = begin + local;
            scratch.coordinates[3U * local] =
                particles.get_positions_x()[index];
            scratch.coordinates[3U * local + 1U] =
                particles.get_positions_y()[index];
            scratch.coordinates[3U * local + 2U] =
                particles.get_positions_z()[index];
        }
        coordinates.update_canonical_real(
            std::span(scratch.coordinates.data(), count * 3U));

        // Hash the exact logical momentum that the native reader must restore,
        // including the writer's p/a rounding and the reader's subsequent *a
        // rounding.  This path is intentionally independent of pack_batch(),
        // so a component/order defect in the transport packer cannot validate
        // itself.
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t index = begin + local;
            scratch.velocities[3U * local] =
                particles.get_momenta_x()[index] * inv_a;
            scratch.velocities[3U * local + 1U] =
                particles.get_momenta_y()[index] * inv_a;
            scratch.velocities[3U * local + 2U] =
                particles.get_momenta_z()[index] * inv_a;
        }
        raw_velocities.update_canonical_real(
            std::span(scratch.velocities.data(), count * 3U));
        for (std::size_t scalar = 0; scalar < count * 3U; ++scalar) {
            scratch.velocities[scalar] *= current_a;
        }
        momenta.update_canonical_real(
            std::span(scratch.velocities.data(), count * 3U));
        ids.update_canonical_uint64(
            particles.get_ids().subspan(begin, count));
        if (include_explicit_masses) {
            masses.update_canonical_real(
                particles.get_masses().subspan(begin, count));
        }
        begin += count;
    }

    const std::array<std::string, snapshot_payload_digest_field_count>
        digests{
            coordinates.finish_hex(),
            raw_velocities.finish_hex(),
            momenta.finish_hex(),
            ids.finish_hex(),
            masses.finish_hex()};
    SnapshotPayloadIdentity identity{};
    for (std::size_t field = 0; field < digests.size(); ++field) {
        if (!is_canonical_sha256(digests[field])) {
            throw std::logic_error(
                "Snapshot payload identity is not canonical SHA-256");
        }
        std::copy(
            digests[field].begin(), digests[field].end(),
            identity.begin()
                + static_cast<std::ptrdiff_t>(
                    field * SHA256_HEX_CHARACTER_COUNT));
    }
    return identity;
}

void update_observed_snapshot_payload_identity(
    const core::ParticleStore& particles,
    bool include_explicit_masses,
    BatchBuffers& scratch,
    Sha256Accumulator& coordinates,
    Sha256Accumulator& momenta,
    Sha256Accumulator& ids,
    Sha256Accumulator& masses) {
    const std::size_t count = particles.num_owned_particles();
    for (std::size_t local = 0; local < count; ++local) {
        scratch.coordinates[3U * local] = particles.get_positions_x()[local];
        scratch.coordinates[3U * local + 1U] =
            particles.get_positions_y()[local];
        scratch.coordinates[3U * local + 2U] =
            particles.get_positions_z()[local];
    }
    coordinates.update_canonical_real(
        std::span(scratch.coordinates.data(), count * 3U));
    for (std::size_t local = 0; local < count; ++local) {
        scratch.velocities[3U * local] = particles.get_momenta_x()[local];
        scratch.velocities[3U * local + 1U] =
            particles.get_momenta_y()[local];
        scratch.velocities[3U * local + 2U] =
            particles.get_momenta_z()[local];
    }
    momenta.update_canonical_real(
        std::span(scratch.velocities.data(), count * 3U));
    ids.update_canonical_uint64(particles.get_ids().first(count));
    if (include_explicit_masses) {
        masses.update_canonical_real(particles.get_masses().first(count));
    }
}

SnapshotPayloadIdentity finish_snapshot_payload_identity(
    Sha256Accumulator& coordinates,
    Sha256Accumulator& raw_velocities,
    Sha256Accumulator& momenta,
    Sha256Accumulator& ids,
    Sha256Accumulator& masses) {
    const std::array<std::string, snapshot_payload_digest_field_count>
        digests{
            coordinates.finish_hex(),
            raw_velocities.finish_hex(),
            momenta.finish_hex(),
            ids.finish_hex(),
            masses.finish_hex()};
    SnapshotPayloadIdentity identity{};
    for (std::size_t field = 0; field < digests.size(); ++field) {
        if (!is_canonical_sha256(digests[field])) {
            throw std::logic_error(
                "Snapshot payload identity is not canonical SHA-256");
        }
        std::copy(
            digests[field].begin(), digests[field].end(),
            identity.begin()
                + static_cast<std::ptrdiff_t>(
                    field * SHA256_HEX_CHARACTER_COUNT));
    }
    return identity;
}

void validate_local_payload(
    const core::ParticleStore& particles,
    core::Real current_a,
    core::Real box_size) {
    if (!std::isfinite(current_a) || current_a <= 0.0) {
        throw std::invalid_argument(
            "Parallel snapshot scale factor must be finite and positive");
    }
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "Parallel snapshot box size must be finite and positive");
    }
    const core::Real inv_a = 1.0 / current_a;
    if (!std::isfinite(inv_a) || inv_a <= 0.0) {
        throw std::overflow_error(
            "Parallel snapshot inverse scale factor is invalid");
    }

    const std::size_t local_n = particles.num_owned_particles();
    for (std::size_t i = 0; i < local_n; ++i) {
        const core::Real x = particles.get_positions_x()[i];
        const core::Real y = particles.get_positions_y()[i];
        const core::Real z = particles.get_positions_z()[i];
        const core::Real px = particles.get_momenta_x()[i];
        const core::Real py = particles.get_momenta_y()[i];
        const core::Real pz = particles.get_momenta_z()[i];
        const core::Real mass = particles.mass_at(i);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
            || !std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) {
            throw std::invalid_argument(
                "Parallel snapshot phase-space fields must be finite");
        }
        if (x < 0.0 || x >= box_size
            || y < 0.0 || y >= box_size
            || z < 0.0 || z >= box_size) {
            throw std::invalid_argument(
                "Parallel snapshot coordinates must lie in periodic [0,L)");
        }
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument(
                "Parallel snapshot particle masses must be finite and positive");
        }
        if (!std::isfinite(px * inv_a)
            || !std::isfinite(py * inv_a)
            || !std::isfinite(pz * inv_a)) {
            throw std::overflow_error(
                "Parallel snapshot momentum-to-velocity conversion overflowed");
        }
    }
}

struct CollectiveMassRepresentation {
    bool uniform{false};
    double uniform_mass{0.0};
};

void require_stage_hard_link(
    hid_t location,
    const std::string& name,
    const char* role) {
    H5L_info_t info{};
    if (H5Lget_info(location, name.c_str(), &info, H5P_DEFAULT) < 0) {
        throw std::runtime_error(
            std::string("Failed to inspect distributed snapshot ")
            + role + " link: " + name);
    }
    if (info.type != H5L_TYPE_HARD) {
        throw std::runtime_error(
            std::string("Distributed snapshot ") + role
            + " must be a file-local hard link: " + name);
    }
}

H5GroupHandle open_stage_group(
    hid_t file,
    const std::string& name) {
    require_stage_hard_link(file, name, "group");
    return H5GroupHandle::checked(
        H5Gopen2(file, name.c_str(), H5P_DEFAULT),
        "open distributed snapshot stage group " + name);
}

template <typename T>
void require_stage_scalar_attr(
    hid_t location,
    const std::string& name,
    hid_t native_type,
    const T& expected) {
    auto attribute = H5AttributeHandle::checked(
        H5Aopen(location, name.c_str(), H5P_DEFAULT),
        "open distributed snapshot stage attribute " + name);
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute.get()),
        "distributed snapshot stage attribute space " + name);
    if (H5Sget_simple_extent_type(space.get()) != H5S_SCALAR) {
        throw std::runtime_error(
            "Distributed snapshot stage attribute must be scalar: " + name);
    }
    require_numeric_attribute_representation(
        attribute.get(), native_type,
        "distributed snapshot stage attribute " + name);
    T observed{};
    check_hdf5(
        H5Aread(
            attribute.get(), native_type, &observed,
            "distributed snapshot stage attribute " + name),
        "read distributed snapshot stage attribute " + name);
    if (observed != expected) {
        throw std::runtime_error(
            "Distributed snapshot stage attribute mismatch: " + name);
    }
}

template <typename T, std::size_t N>
void require_stage_array_attr(
    hid_t location,
    const std::string& name,
    hid_t native_type,
    const std::array<T, N>& expected) {
    auto attribute = H5AttributeHandle::checked(
        H5Aopen(location, name.c_str(), H5P_DEFAULT),
        "open distributed snapshot stage array attribute " + name);
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute.get()),
        "distributed snapshot stage array attribute space " + name);
    if (H5Sget_simple_extent_ndims(space.get()) != 1) {
        throw std::runtime_error(
            "Distributed snapshot stage array attribute must have rank one: "
            + name);
    }
    hsize_t dimensions[1]{0};
    check_hdf5(
        H5Sget_simple_extent_dims(space.get(), dimensions, nullptr),
        "read distributed snapshot stage array dimensions " + name);
    if (dimensions[0] != static_cast<hsize_t>(N)) {
        throw std::runtime_error(
            "Distributed snapshot stage array attribute length mismatch: "
            + name);
    }
    require_numeric_attribute_representation(
        attribute.get(), native_type,
        "distributed snapshot stage array attribute " + name);
    std::array<T, N> observed{};
    check_hdf5(
        H5Aread(
            attribute.get(), native_type, observed.data(),
            "distributed snapshot stage array attribute " + name),
        "read distributed snapshot stage array attribute " + name);
    if (observed != expected) {
        throw std::runtime_error(
            "Distributed snapshot stage array attribute mismatch: " + name);
    }
}

void require_stage_string_attr(
    hid_t location,
    const std::string& name,
    const std::string& expected) {
    const std::string observed = read_bounded_fixed_string_attribute(
        location,
        name,
        expected.size(),
        false,
        "Distributed snapshot stage");
    if (observed != expected) {
        throw std::runtime_error(
            "Distributed snapshot stage string attribute mismatch: " + name);
    }
}

SnapshotDescriptor validate_stage_metadata(
    const VerifiedSnapshotSource& source,
    const config::SimulationParameters& config,
    core::Real current_a,
    std::uint64_t global_count,
    const CollectiveMassRepresentation& mass_representation,
    const std::string& physics_fingerprint,
    const std::string& metadata_json,
    const RunMetadata& run_metadata) {
    auto file = H5FileHandle::checked(
        H5Fopen(
            source.hdf5_read_path().c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT),
        "reopen distributed snapshot stage for metadata admission");
    auto header = open_stage_group(file.get(), schema::GROUP_HEADER);
    auto parameters = open_stage_group(file.get(), schema::GROUP_PARAMETERS);
    auto config_group = open_stage_group(file.get(), schema::GROUP_CONFIG);
    require_stage_hard_link(
        file.get(), schema::GROUP_PART_TYPE_1, "group");

    const std::uint32_t count = static_cast<std::uint32_t>(global_count);
    const std::array<std::uint32_t, 6> particle_counts = {
        0U, count, 0U, 0U, 0U, 0U};
    std::array<double, 6> mass_table = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (mass_representation.uniform) {
        mass_table[1] = mass_representation.uniform_mass;
    }
    const double scale_factor = static_cast<double>(current_a);
    const double redshift = 1.0 / scale_factor - 1.0;
    const double box_size = static_cast<double>(config.get_box().L);
    const double omega_m = config.get_cosmology().omega_m;
    const double omega_lambda = config.get_cosmology().omega_lambda;
    const double hubble_param = config.get_cosmology().h;
    const std::int32_t one = 1;
    const std::int32_t double_precision = sizeof(core::Real) == 8 ? 1 : 0;
    const double unit_length = cosmology::units::UnitLength_in_cm;
    const double unit_mass = cosmology::units::UnitMass_in_g;
    const double unit_velocity = cosmology::units::UnitVelocity_in_cm_s;

    require_stage_array_attr(
        header.get(), schema::ATTR_NUM_PART_THIS_FILE,
        H5T_NATIVE_UINT32, particle_counts);
    require_stage_array_attr(
        header.get(), schema::ATTR_NUM_PART_TOTAL,
        H5T_NATIVE_UINT32, particle_counts);
    require_stage_array_attr(
        header.get(), schema::ATTR_MASS_TABLE,
        H5T_NATIVE_DOUBLE, mass_table);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_TIME,
        H5T_NATIVE_DOUBLE, scale_factor);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_REDSHIFT,
        H5T_NATIVE_DOUBLE, redshift);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_BOX_SIZE,
        H5T_NATIVE_DOUBLE, box_size);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_OMEGA0,
        H5T_NATIVE_DOUBLE, omega_m);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_OMEGA_LAMBDA,
        H5T_NATIVE_DOUBLE, omega_lambda);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_HUBBLE_PARAM,
        H5T_NATIVE_DOUBLE, hubble_param);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_NUM_FILES,
        H5T_NATIVE_INT32, one);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_FLAG_DOUBLE_PRECISION,
        H5T_NATIVE_INT32, double_precision);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_UNIT_LENGTH_IN_CM,
        H5T_NATIVE_DOUBLE, unit_length);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_UNIT_MASS_IN_G,
        H5T_NATIVE_DOUBLE, unit_mass);
    require_stage_scalar_attr(
        header.get(), schema::ATTR_UNIT_VELOCITY_IN_CM_PER_S,
        H5T_NATIVE_DOUBLE, unit_velocity);

    const std::uint64_t seed = run_metadata.seed;
    const std::uint32_t particles_per_dimension =
        static_cast<std::uint32_t>(config.get_box().N);
    const std::uint32_t pm_mesh_per_dimension =
        static_cast<std::uint32_t>(config.get_box().N_mesh);
    const std::uint32_t ic_mesh_per_dimension =
        static_cast<std::uint32_t>(run_metadata.ic_mesh_per_dimension);
    const double softening = static_cast<double>(config.eps());
    const double sigma8 = config.get_cosmology().sigma8;
    const double spectral_index = config.get_cosmology().n_s;
    const double omega_b = config.get_cosmology().omega_b;
    const std::int32_t lpt_order = run_metadata.lpt_order;
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_SEED,
        H5T_NATIVE_UINT64, seed);
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_PARTICLES_PER_DIM,
        H5T_NATIVE_UINT32, particles_per_dimension);
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_PM_MESH_PER_DIM,
        H5T_NATIVE_UINT32, pm_mesh_per_dimension);
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_IC_MESH_PER_DIM,
        H5T_NATIVE_UINT32, ic_mesh_per_dimension);
    require_stage_string_attr(
        parameters.get(), schema::ATTR_IC_AMPLITUDE_MODE,
        run_metadata.ic_amplitude_mode);
    require_stage_string_attr(
        parameters.get(), schema::ATTR_IC_PHASE_PAIRING,
        run_metadata.ic_phase_pairing);
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_SOFTENING_COMOVING,
        H5T_NATIVE_DOUBLE, softening);
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_SIGMA8,
        H5T_NATIVE_DOUBLE, sigma8);
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_N_S,
        H5T_NATIVE_DOUBLE, spectral_index);
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_OMEGA_B,
        H5T_NATIVE_DOUBLE, omega_b);
    require_stage_scalar_attr(
        parameters.get(), schema::ATTR_LPT_ORDER,
        H5T_NATIVE_INT32, lpt_order);
    require_stage_string_attr(
        parameters.get(), schema::ATTR_POWER_SPECTRUM_FIDELITY,
        run_metadata.power_spectrum_fidelity);

    require_stage_string_attr(
        config_group.get(), schema::ATTR_PHYSICS_FINGERPRINT,
        physics_fingerprint);
    require_stage_string_attr(
        config_group.get(), schema::ATTR_RUN_METADATA_JSON,
        metadata_json);
    for (const auto& binding : schema::NATIVE_SNAPSHOT_SEMANTIC_BINDINGS) {
        require_stage_string_attr(
            config_group.get(),
            std::string(binding.attribute),
            std::string(binding.value));
    }

    SnapshotDescriptor descriptor;
    descriptor.scale_factor = current_a;
    descriptor.redshift = static_cast<core::Real>(redshift);
    descriptor.box_size_Mpc_h = config.get_box().L;
    descriptor.omega_m = static_cast<core::Real>(omega_m);
    descriptor.omega_lambda = static_cast<core::Real>(omega_lambda);
    descriptor.omega_b = static_cast<core::Real>(omega_b);
    descriptor.hubble_param = static_cast<core::Real>(hubble_param);
    descriptor.particle_count = global_count;
    descriptor.uniform_mass = mass_representation.uniform;
    descriptor.uniform_particle_mass = static_cast<core::Real>(
        mass_representation.uniform_mass);
    descriptor.particles_per_dimension = config.get_box().N;
    descriptor.pm_mesh_per_dimension = config.get_box().N_mesh;
    descriptor.ic_mesh_per_dimension = run_metadata.ic_mesh_per_dimension;
    descriptor.ic_seed = seed;
    descriptor.lpt_order = lpt_order;
    descriptor.softening_comoving_Mpc_h = config.eps();
    descriptor.sigma8 = static_cast<core::Real>(sigma8);
    descriptor.spectral_index_ns = static_cast<core::Real>(spectral_index);
    descriptor.ic_amplitude_mode = run_metadata.ic_amplitude_mode;
    descriptor.ic_phase_pairing = run_metadata.ic_phase_pairing;
    descriptor.power_spectrum_fidelity =
        run_metadata.power_spectrum_fidelity;
    descriptor.generation_provenance =
        read_snapshot_generation_provenance(metadata_json);
    descriptor.physics_fingerprint = physics_fingerprint;
    descriptor.run_metadata_json = metadata_json;
    descriptor.native_snapshot_object_sha256 = source.sha256();
    for (std::size_t index = 0;
         index < schema::NATIVE_SNAPSHOT_SEMANTIC_BINDINGS.size();
         ++index) {
        descriptor.semantic_binding_values[index] = std::string(
            schema::NATIVE_SNAPSHOT_SEMANTIC_BINDINGS[index].value);
    }
    return descriptor;
}

CollectiveMassRepresentation require_collective_mass_representation(
    const core::ParticleStore& particles) {
    const std::size_t local_count = particles.num_owned_particles();
    const auto local_uniform_mass = particles.get_uniform_mass();

    // An empty rank owns no samples of the mass field, so its local storage
    // representation is mathematically neutral. Only ranks that own particles
    // may constrain whether the global field is uniform or explicit.
    const int local_uniform_owner =
        local_count != 0 && local_uniform_mass.has_value() ? 1 : 0;
    const int local_explicit_owner =
        local_count != 0 && !local_uniform_mass.has_value() ? 1 : 0;
    int any_uniform_owner = 0;
    int any_explicit_owner = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_uniform_owner, &any_uniform_owner, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for parallel snapshot uniform-mass owners");
    require_mpi_success(
        MPI_Allreduce(
            &local_explicit_owner, &any_explicit_owner, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for parallel snapshot explicit-mass owners");
    if (any_uniform_owner != 0 && any_explicit_owner != 0) {
        throw std::invalid_argument(
            "Parallel snapshot populated ranks use mixed mass representations");
    }
    if (any_uniform_owner == 0) {
        // This covers explicit-mass snapshots and the degenerate globally empty
        // snapshot, for which there is no physical mass sample to encode.
        return {};
    }

    const double local_mass = local_uniform_owner != 0
        ? static_cast<double>(*local_uniform_mass)
        : 0.0;
    const int local_invalid = local_uniform_owner != 0
        && (!std::isfinite(local_mass) || local_mass <= 0.0) ? 1 : 0;
    int any_invalid = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_invalid, &any_invalid, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for parallel snapshot uniform-mass validity");
    if (any_invalid != 0) {
        throw std::invalid_argument(
            "Parallel snapshot uniform mass must be finite and positive on every populated rank");
    }

    double local_min = local_uniform_owner != 0
        ? local_mass : std::numeric_limits<double>::max();
    double local_max = local_uniform_owner != 0
        ? local_mass : std::numeric_limits<double>::lowest();
    double mass_min = 0.0;
    double mass_max = 0.0;
    require_mpi_success(
        MPI_Allreduce(
            &local_min, &mass_min, 1,
            MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce for minimum parallel snapshot uniform mass");
    require_mpi_success(
        MPI_Allreduce(
            &local_max, &mass_max, 1,
            MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for maximum parallel snapshot uniform mass");
    if (mass_min != mass_max) {
        throw std::invalid_argument(
            "Parallel snapshot uniform mass differs across populated MPI ranks");
    }
    return CollectiveMassRepresentation{true, mass_min};
}

void pack_batch(
    const core::ParticleStore& particles,
    core::Real inv_a,
    std::size_t begin,
    std::size_t count,
    bool include_masses,
    BatchBuffers& buffers) {
    for (std::size_t local = 0; local < count; ++local) {
        const std::size_t i = begin + local;
        buffers.coordinates[3 * local + 0] = particles.get_positions_x()[i];
        buffers.coordinates[3 * local + 1] = particles.get_positions_y()[i];
        buffers.coordinates[3 * local + 2] = particles.get_positions_z()[i];
        buffers.velocities[3 * local + 0] = particles.get_momenta_x()[i] * inv_a;
        buffers.velocities[3 * local + 1] = particles.get_momenta_y()[i] * inv_a;
        buffers.velocities[3 * local + 2] = particles.get_momenta_z()[i] * inv_a;
        buffers.ids[local] = particles.get_ids()[i];
        if (include_masses) buffers.masses[local] = particles.get_masses()[i];
    }
}

MPI_Datatype mpi_real_datatype() {
    static_assert(
        sizeof(core::Real) == sizeof(double)
        || sizeof(core::Real) == sizeof(float));
    if constexpr (sizeof(core::Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    return MPI_FLOAT;
}

#endif

} // namespace

void ParallelSnapshotIO::write_snapshot_payload(
    const core::ParticleStore& local_particles,
    core::Real current_a,
    int snapshot_index) const {
#ifndef COSMO_NBODY_HAS_MPI
    (void)local_particles;
    (void)current_a;
    (void)snapshot_index;
    throw std::runtime_error(
        "Distributed snapshot output requires an MPI-enabled build");
#else
#include "parallel_snapshot_preflight.inc"
#include "parallel_snapshot_hdf5_payload.inc"
#endif
}

} // namespace io
} // namespace cosmo_nbody
