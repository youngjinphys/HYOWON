#include "cosmo_nbody/io/restart_io.hpp"
#include "cosmo_nbody/io/artifact_schema.hpp"
#include "bounded_text_serialization.hpp"
#include "cosmo_nbody/io/bounded_hdf5_string.hpp"
#include "cosmo_nbody/io/content_hash.hpp"
#include "cosmo_nbody/io/durable_file_publication.hpp"
#include "cosmo_nbody/io/hdf5_handle.hpp"
#include "cosmo_nbody/io/native_identity.hpp"
#include "cosmo_nbody/io/persisted_text_limits.hpp"
#include "cosmo_nbody/io/restart_state_identity.hpp"
#include "cosmo_nbody/io/runtime_provenance.hpp"
#include "cosmo_nbody/core/id_uniqueness.hpp"
#include "cosmo_nbody/runtime/mpi_collective_stage.hpp"

#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#ifdef COSMO_NBODY_HAS_MPI
#include <mpi.h>
#endif

namespace cosmo_nbody {
namespace io {

RestartIO::RestartIO(const config::SimulationParameters& config)
    : config_(config) {}

namespace {

constexpr const char* VERIFIED_SNAPSHOT_IC_SHA256 =
    "VerifiedSnapshotICSHA256";
constexpr const char* RESTART_PRODUCT_KIND_ATTRIBUTE = "product_kind";
constexpr const char* RESTART_PRODUCT_KIND = "serial_restart";
constexpr const char* RESTART_DYNAMICS_SHA256_ATTRIBUTE =
    "RestartDynamicsSHA256";
constexpr const char* RESTART_STATE_SHA256_ATTRIBUTE =
    "RestartStateSHA256";

std::filesystem::path normalized_restart_path(const std::string& filename) {
    std::filesystem::path path(filename);
    if (path.extension() != ".hdf5") path += ".hdf5";
    return path;
}

void write_attr(hid_t loc, const std::string& name, hid_t type, const void* data) {
    auto space = H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR), "restart attribute space " + name);
    auto attr = H5AttributeHandle::checked(
        H5Acreate2(loc, name.c_str(), type, space.get(), H5P_DEFAULT, H5P_DEFAULT),
        "restart attribute " + name);
    check_hdf5(H5Awrite(attr.get(), type, data),
               "write restart attribute " + name);
}

void require_scalar_attr(hid_t attribute, const std::string& name) {
    auto space = H5SpaceHandle::checked(
        H5Aget_space(attribute), "restart attribute dataspace " + name);
    if (H5Sget_simple_extent_type(space.get()) != H5S_SCALAR) {
        throw std::runtime_error(
            "Restart attribute must be scalar: " + name);
    }
}

void read_attr(hid_t loc, const std::string& name, hid_t type, void* data) {
    auto attr = H5AttributeHandle::checked(
        H5Aopen(loc, name.c_str(), H5P_DEFAULT), "restart attribute " + name);
    require_scalar_attr(attr.get(), name);
    check_hdf5(H5Aread(attr.get(), type, data),
               "read restart attribute " + name);
}

void write_string_attr(hid_t loc, const std::string& name, const std::string& value) {
    auto space = H5SpaceHandle::checked(
        H5Screate(H5S_SCALAR), "restart string space " + name);
    auto type = H5TypeHandle::checked(
        H5Tcopy(H5T_C_S1), "restart string type " + name);
    check_hdf5(H5Tset_size(type.get(), value.size() + 1),
               "set restart string size " + name);
    check_hdf5(H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
               "set restart string padding " + name);
    auto attr = H5AttributeHandle::checked(
        H5Acreate2(loc, name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT),
        "restart string attribute " + name);
    check_hdf5(H5Awrite(attr.get(), type.get(), value.c_str()),
               "write restart string " + name);
}

std::string read_string_attr(
    hid_t loc,
    const std::string& name,
    std::size_t maximum_logical_bytes) {
    return read_bounded_fixed_string_attribute(
        loc,
        name,
        maximum_logical_bytes,
        false,
        "Restart");
}

H5DatasetHandle create_dataset(
    hid_t group,
    const std::string& name,
    hid_t type,
    hid_t space) {
    return H5DatasetHandle::checked(
        H5Dcreate2(group, name.c_str(), type, space,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        "restart dataset " + name);
}

H5DatasetHandle open_dataset(hid_t group, const std::string& name) {
    return H5DatasetHandle::checked(
        H5Dopen2(group, name.c_str(), H5P_DEFAULT),
        "restart dataset " + name);
}

void require_length(hid_t dataset, const std::string& name, std::size_t expected) {
    auto space = H5SpaceHandle::checked(
        H5Dget_space(dataset), "restart dataspace " + name);
    if (H5Sget_simple_extent_ndims(space.get()) != 1) {
        throw std::runtime_error("Restart dataset must have rank 1: " + name);
    }
    hsize_t dims[1] = {0};
    check_hdf5(H5Sget_simple_extent_dims(space.get(), dims, nullptr),
               "read restart dimensions " + name);
    if (dims[0] != static_cast<hsize_t>(expected)) {
        throw std::runtime_error("Restart dataset length mismatch: " + name);
    }
}

bool canonical_sha256(std::string_view value) noexcept {
    if (value.size() != 64U) return false;
    for (const char character : value) {
        const bool decimal = character >= '0' && character <= '9';
        const bool lower_hex = character >= 'a' && character <= 'f';
        if (!decimal && !lower_hex) return false;
    }
    return true;
}

void require_valid_restart_payload(const core::ParticleStore& particles) {
    const std::size_t n = particles.num_owned_particles();
    if constexpr (sizeof(std::size_t) > sizeof(unsigned long long)) {
        if (n > static_cast<std::size_t>(
                std::numeric_limits<unsigned long long>::max())) {
            throw std::overflow_error(
                "Restart particle count exceeds unsigned long long range");
        }
    }
    if (const auto uniform_mass = particles.get_uniform_mass();
        uniform_mass.has_value()
        && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
        throw std::invalid_argument(
            "Restart uniform particle mass must be finite and positive");
    }
    for (std::size_t i = 0; i < n; ++i) {
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
                "Restart particle phase-space fields must be finite");
        }
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument(
                "Restart particle masses must be finite and positive");
        }
    }
    core::require_unique_particle_ids_bounded(
        particles.get_ids().first(n), "Restart payload");
}

void require_valid_restart_stepper(const time::TimeStepper& stepper) {
    if (!std::isfinite(stepper.current_a()) || stepper.current_a() <= 0.0) {
        throw std::invalid_argument(
            "Restart TimeStepper current scale factor must be finite and positive");
    }
    if (!std::isfinite(stepper.delta_ln_a()) || stepper.delta_ln_a() <= 0.0) {
        throw std::invalid_argument(
            "Restart TimeStepper delta_ln_a must be finite and positive");
    }
    if (stepper.current_step() > stepper.num_steps()) {
        throw std::invalid_argument(
            "Restart TimeStepper step index exceeds boundary sequence");
    }
}

struct RestartMpiTopology {
    int rank{0};
    int size{1};
};

RestartMpiTopology restart_mpi_topology(
    const config::SimulationParameters& config) {
    if (!config.get_runtime().mpi_enabled) return RestartMpiTopology{};
#ifdef COSMO_NBODY_HAS_MPI
    runtime::require_active_mpi_main_thread("Restart identity topology");
    int rank = 0;
    int size = 0;
    const int rank_status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank_status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, rank_status);
        std::abort();
    }
    const int size_status = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size_status != MPI_SUCCESS) {
        (void)MPI_Abort(MPI_COMM_WORLD, size_status);
        std::abort();
    }
    if (rank < 0 || size < 1 || rank >= size) {
        throw std::runtime_error(
            "Failed to query MPI topology for restart identity");
    }
    return RestartMpiTopology{rank, size};
#else
    throw std::runtime_error(
        "Restart identity requested MPI in a non-MPI build");
#endif
}

int restart_mpi_size(const config::SimulationParameters& config) {
    return restart_mpi_topology(config).size;
}

std::string restart_dynamics_descriptor(
    const config::SimulationParameters& config,
    const std::string& verified_snapshot_ic_sha256,
    bool include_legacy_input_paths) {
    return detail::render_bounded_text(
        MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES,
        [&](std::ostream& out) {
            out << "product_kind=serial_restart_dynamics\n";

            auto append_string = [&](
                const char* key,
                const std::string& value) {
                out << key << '=' << value.size() << ':' << value << '\n';
            };
            auto append_real = [&](const char* key, core::Real value) {
                out << key << '=' << value << '\n';
            };
            auto append_uint = [&](
                const char* key,
                std::uint64_t value) {
                out << key << '=' << value << '\n';
            };
            auto append_bool = [&](const char* key, bool value) {
                out << key << '=' << (value ? 1 : 0) << '\n';
            };

            const auto& c = config.get_cosmology();
            append_real("h", c.h);
            append_real("omega_m", c.omega_m);
            append_real("omega_lambda", c.omega_lambda);
            append_real("omega_b", c.omega_b);
            append_real("sigma8", c.sigma8);
            append_real("n_s", c.n_s);

            const auto& b = config.get_box();
            append_real("box_L", b.L);
            append_uint("box_N", b.N);
            append_uint("box_N_mesh", b.N_mesh);

            const auto& g = config.get_gravity();
            append_string("gravity_solver", g.solver);
            append_real("gravity_eps", config.eps());
            append_real("gravity_theta", g.theta);
            append_real("gravity_split_scale_cells", g.split_scale_cells);
            append_bool("gravity_deconvolve_cic", g.deconvolve_cic);
            append_real("gravity_r_s", config.r_s());
            append_real("gravity_r_cut", config.r_cut());

            const auto& t = config.get_time();
            append_real("time_z_start", t.z_start);
            append_real("time_z_final", t.z_final);
            append_real("time_delta_ln_a", t.delta_ln_a);
            append_string("time_step_policy", t.step_policy);

            const auto& ic = config.get_ic();
            append_string("ic_mode", ic.mode);
            append_uint(
                "ic_lpt_order", static_cast<std::uint64_t>(ic.lpt_order));
            append_uint("ic_seed", ic.seed);
            append_uint(
                "ic_mesh_per_dimension", config.ic_mesh_per_dimension());
            append_string("ic_amplitude_mode", ic.amplitude_mode);
            append_string("ic_phase_pairing", ic.phase_pairing);
            if (include_legacy_input_paths) {
                append_string("ic_power_spectrum_file", ic.power_spectrum_file);
            }
            append_string(
                "ic_power_spectrum_sha256", ic.power_spectrum_sha256);
            append_real(
                "ic_power_spectrum_redshift", ic.power_spectrum_redshift);
            append_string(
                "ic_power_spectrum_fidelity", ic.power_spectrum_fidelity);
            if (include_legacy_input_paths) {
                append_string("ic_snapshot_file", ic.snapshot_file);
            }
            append_string(
                "ic_snapshot_sha256",
                resolve_verified_snapshot_ic_sha256(
                    config, verified_snapshot_ic_sha256));

            const auto& output = config.get_output();
            append_uint(
                "snapshot_target_count",
                static_cast<std::uint64_t>(
                    output.snapshot_scale_factors.size()));
            for (const core::Real target : output.snapshot_scale_factors) {
                append_real("snapshot_target", target);
            }

            const auto& runtime = config.get_runtime();
            append_bool("runtime_mpi_enabled", runtime.mpi_enabled);
            append_uint(
                "runtime_mpi_size",
                static_cast<std::uint64_t>(restart_mpi_size(config)));
        });
}

std::string restart_dynamics_sha256(
    const config::SimulationParameters& config,
    const std::string& verified_snapshot_ic_sha256,
    const artifact_schema::RestartSchema& encoding) {
    return sha256_text(restart_dynamics_descriptor(
        config,
        verified_snapshot_ic_sha256,
        encoding.includes_input_paths));
}

std::uint64_t restart_step_u64(const time::TimeStepper& stepper) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (stepper.current_step()
            > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error("Restart step index exceeds uint64 range");
        }
    }
    return static_cast<std::uint64_t>(stepper.current_step());
}

std::string restart_state_sha256(
    const config::SimulationParameters& config,
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper,
    std::string_view dynamics_identity) {
    const RestartMpiTopology topology = restart_mpi_topology(config);
    const std::size_t count = particles.num_owned_particles();
    const bool uniform_mass = particles.get_uniform_mass().has_value();
    return sha256_restart_state(RestartStateIdentityInput{
        .rank = topology.rank,
        .ranks = topology.size,
        .step = restart_step_u64(stepper),
        .current_a = stepper.current_a(),
        .delta_ln_a = stepper.delta_ln_a(),
        .dynamics_sha256 = dynamics_identity,
        .positions_x = particles.get_positions_x().first(count),
        .positions_y = particles.get_positions_y().first(count),
        .positions_z = particles.get_positions_z().first(count),
        .momenta_x = particles.get_momenta_x().first(count),
        .momenta_y = particles.get_momenta_y().first(count),
        .momenta_z = particles.get_momenta_z().first(count),
        .ids = particles.get_ids().first(count),
        .uniform_mass = particles.get_uniform_mass(),
        .masses = uniform_mass
            ? std::span<const core::Real>{}
            : particles.get_masses().first(count),
    });
}

} // namespace

void RestartIO::write_restart(
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper,
    const std::string& filename) const {
    (void)write_restart_with_identity(particles, stepper, filename);
}

RestartStateIdentityResult RestartIO::write_restart_with_identity(
    const core::ParticleStore& particles,
    const time::TimeStepper& stepper,
    const std::string& filename) const {
    require_valid_restart_payload(particles);
    require_valid_restart_stepper(stepper);

    const std::string verified_snapshot_ic_sha256 =
        resolve_verified_snapshot_ic_sha256(
            config_, particles.get_verified_snapshot_ic_sha256());
    config_.set_verified_snapshot_ic_sha256(verified_snapshot_ic_sha256);

    DurableFilePublication publication(
        normalized_restart_path(filename),
        "restart payload");
    std::string state_sha256;
    {
        const std::string stage = publication.staging_path().string();
        auto file = H5FileHandle::checked(
            H5Fcreate(stage.c_str(), H5F_ACC_EXCL,
                      H5P_DEFAULT, H5P_DEFAULT),
            "create restart " + stage);
        const hid_t type_real = sizeof(core::Real) == 8
            ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

        write_string_attr(
            file.get(),
            native_identity::SOFTWARE_NAME_ATTRIBUTE_STRING,
            native_identity::SOFTWARE_NAME_STRING);
        write_string_attr(
            file.get(), std::string(artifact_schema::RESTART_ATTRIBUTE),
            std::string(artifact_schema::CURRENT_RESTART.identity));
        write_string_attr(
            file.get(), RESTART_PRODUCT_KIND_ATTRIBUTE,
            RESTART_PRODUCT_KIND);

        const std::string dynamics_identity = restart_dynamics_sha256(
            config_, verified_snapshot_ic_sha256,
            artifact_schema::CURRENT_RESTART);
        state_sha256 = restart_state_sha256(
            config_, particles, stepper, dynamics_identity);

        write_string_attr(
            file.get(), RESTART_DYNAMICS_SHA256_ATTRIBUTE,
            dynamics_identity);
        write_string_attr(
            file.get(), VERIFIED_SNAPSHOT_IC_SHA256,
            verified_snapshot_ic_sha256);
        write_string_attr(
            file.get(), RESTART_STATE_SHA256_ATTRIBUTE,
            state_sha256);

        auto step_group = H5GroupHandle::checked(
            H5Gcreate2(file.get(), "/TimeStepper",
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "restart TimeStepper group");
        const double delta_ln_a = static_cast<double>(stepper.delta_ln_a());
        const double current_a = static_cast<double>(stepper.current_a());
        const unsigned long long step =
            static_cast<unsigned long long>(stepper.current_step());
        write_attr(step_group.get(), "Delta_ln_a", H5T_NATIVE_DOUBLE, &delta_ln_a);
        write_attr(step_group.get(), "Current_a", H5T_NATIVE_DOUBLE, &current_a);
        write_attr(step_group.get(), "CurrentStep", H5T_NATIVE_ULLONG, &step);

        auto particle_group = H5GroupHandle::checked(
            H5Gcreate2(file.get(), "/Particles",
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            "restart Particles group");
        const std::size_t n = particles.num_owned_particles();
        const unsigned long long n_ull = static_cast<unsigned long long>(n);
        write_attr(particle_group.get(), "NumParticles",
                   H5T_NATIVE_ULLONG, &n_ull);

        const int uniform = particles.get_uniform_mass().has_value() ? 1 : 0;
        const double uniform_mass = uniform
            ? static_cast<double>(*particles.get_uniform_mass()) : 0.0;
        write_attr(particle_group.get(), "UniformMassFlag", H5T_NATIVE_INT, &uniform);
        write_attr(particle_group.get(), "UniformMass", H5T_NATIVE_DOUBLE, &uniform_mass);

        const hsize_t dims[1] = {static_cast<hsize_t>(n)};
        auto space = H5SpaceHandle::checked(
            H5Screate_simple(1, dims, nullptr), "restart particle space");
        auto write_dataset = [&](const std::string& name, hid_t type, const void* data) {
            auto dataset = create_dataset(
                particle_group.get(), name, type, space.get());
            if (n > 0) {
                check_hdf5(H5Dwrite(dataset.get(), type, H5S_ALL, H5S_ALL,
                                    H5P_DEFAULT, data),
                           "write restart dataset " + name);
            }
        };
        write_dataset("pos_x", type_real, particles.get_positions_x().data());
        write_dataset("pos_y", type_real, particles.get_positions_y().data());
        write_dataset("pos_z", type_real, particles.get_positions_z().data());
        write_dataset("mom_x", type_real, particles.get_momenta_x().data());
        write_dataset("mom_y", type_real, particles.get_momenta_y().data());
        write_dataset("mom_z", type_real, particles.get_momenta_z().data());
        write_dataset("ParticleIDs", H5T_NATIVE_UINT64, particles.get_ids().data());
        if (!uniform) {
            write_dataset("masses", type_real, particles.get_masses().data());
        }
        check_hdf5(H5Fflush(file.get(), H5F_SCOPE_GLOBAL), "flush restart file");
    }
    publication.publish_nonempty();
    return RestartStateIdentityResult{state_sha256, true};
}

void RestartIO::read_restart(
    const std::string& filename,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out) const {
    (void)read_restart_with_identity(
        filename, particles_out, stepper_out);
}

void RestartIO::read_restart(
    RestartReadSession& session,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out) const {
    (void)read_restart_with_identity(
        session, particles_out, stepper_out);
}

RestartStateIdentityResult RestartIO::read_restart_with_identity(
    const std::string& filename,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out,
    std::string_view expected_state_sha256) const {
    auto session = open_restart_session(filename);
    return read_restart_with_identity(
        session, particles_out, stepper_out, expected_state_sha256);
}

RestartStateIdentityResult RestartIO::read_restart_with_identity(
    RestartReadSession& session,
    core::ParticleStore& particles_out,
    time::TimeStepper& stepper_out,
    std::string_view expected_state_sha256) const {
    const hid_t file = session.file_id();
    if (file < 0) {
        throw std::invalid_argument(
            "Restart read requires an open session");
    }
    if (!expected_state_sha256.empty()
        && !canonical_sha256(expected_state_sha256)) {
        throw std::invalid_argument(
            "Expected restart-state SHA-256 must be canonical lowercase hexadecimal");
    }
    const hid_t type_real = sizeof(core::Real) == 8
        ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

    const std::string software_name = read_string_attr(
        file,
        native_identity::SOFTWARE_NAME_ATTRIBUTE_STRING,
        native_identity::SOFTWARE_NAME.size());
    if (software_name != native_identity::SOFTWARE_NAME_STRING) {
        throw std::runtime_error(
            "Restart SoftwareName is not the HYOWON native identity");
    }
    const std::string restart_schema = read_string_attr(
        file,
        std::string(artifact_schema::RESTART_ATTRIBUTE),
        artifact_schema::MAXIMUM_RESTART_SCHEMA_BYTES);
    const auto* encoding = artifact_schema::lookup_restart(restart_schema);
    if (encoding == nullptr) {
        throw std::runtime_error("Restart HYOWON schema is unsupported");
    }
    const std::string product_kind = read_string_attr(
        file, RESTART_PRODUCT_KIND_ATTRIBUTE,
        std::string_view(RESTART_PRODUCT_KIND).size());
    if (product_kind != RESTART_PRODUCT_KIND) {
        throw std::runtime_error("Restart product kind is unsupported");
    }

    const std::string stored_snapshot_ic_sha256 = read_string_attr(
        file, VERIFIED_SNAPSHOT_IC_SHA256, 64U);
    const std::string verified_snapshot_ic_sha256 =
        resolve_verified_snapshot_ic_sha256(
            config_, stored_snapshot_ic_sha256);

    const std::string stored_dynamics_identity = read_string_attr(
        file, RESTART_DYNAMICS_SHA256_ATTRIBUTE, 64U);
    if (!canonical_sha256(stored_dynamics_identity)) {
        throw std::runtime_error(
            "Restart dynamics identity is malformed");
    }
    if (stored_dynamics_identity
        != restart_dynamics_sha256(
            config_,
            verified_snapshot_ic_sha256,
            *encoding)) {
        throw std::runtime_error("Restart dynamics identity mismatch");
    }

    const std::string stored_state_sha256 = read_string_attr(
        file, RESTART_STATE_SHA256_ATTRIBUTE, 64U);
    if (!canonical_sha256(stored_state_sha256)) {
        throw std::runtime_error(
            "Restart state identity digest is malformed");
    }

    auto step_group = H5GroupHandle::checked(
        H5Gopen2(file, "/TimeStepper", H5P_DEFAULT),
        "restart TimeStepper group");
    unsigned long long step = 0;
    double current_a_disk = 0.0;
    double delta_ln_a_disk = 0.0;
    read_attr(step_group.get(), "CurrentStep", H5T_NATIVE_ULLONG, &step);
    read_attr(step_group.get(), "Current_a", H5T_NATIVE_DOUBLE, &current_a_disk);
    read_attr(step_group.get(), "Delta_ln_a", H5T_NATIVE_DOUBLE, &delta_ln_a_disk);

    if (!std::isfinite(current_a_disk) || current_a_disk <= 0.0) {
        throw std::runtime_error(
            "Restart current scale factor must be finite and positive");
    }
    if (!std::isfinite(delta_ln_a_disk) || delta_ln_a_disk <= 0.0) {
        throw std::runtime_error(
            "Restart delta_ln_a must be finite and positive");
    }
    if (static_cast<core::Real>(delta_ln_a_disk)
        != stepper_out.delta_ln_a()) {
        throw std::runtime_error(
            "Restart delta_ln_a does not exactly match TimeStepper");
    }
    if (step > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Restart step index does not fit size_t");
    }

    auto particle_group = H5GroupHandle::checked(
        H5Gopen2(file, "/Particles", H5P_DEFAULT),
        "restart Particles group");
    unsigned long long n_ull = 0;
    int uniform_flag = 0;
    double uniform_mass = 0.0;
    read_attr(particle_group.get(), "NumParticles", H5T_NATIVE_ULLONG, &n_ull);
    read_attr(particle_group.get(), "UniformMassFlag", H5T_NATIVE_INT, &uniform_flag);
    read_attr(particle_group.get(), "UniformMass", H5T_NATIVE_DOUBLE, &uniform_mass);

    if (n_ull > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Restart particle count does not fit size_t");
    }
    if (uniform_flag != 0 && uniform_flag != 1) {
        throw std::runtime_error("Restart UniformMassFlag is invalid");
    }
    if (!std::isfinite(uniform_mass) || uniform_mass < 0.0) {
        throw std::runtime_error(
            "Restart UniformMass must be finite and non-negative");
    }
    if (uniform_flag == 1 && uniform_mass <= 0.0) {
        throw std::runtime_error(
            "Restart uniform mass must be finite and positive");
    }

    const std::size_t n = static_cast<std::size_t>(n_ull);
    core::ParticleStore loaded;
    if (uniform_flag == 1) {
        loaded.set_uniform_mass(static_cast<core::Real>(uniform_mass));
    } else {
        loaded.set_uniform_mass(std::nullopt);
    }
    loaded.resize(n);

    if (n > 0) {
        auto read_dataset = [&](const std::string& name, hid_t type, void* data) {
            auto dataset = open_dataset(particle_group.get(), name);
            require_length(dataset.get(), name, n);
            check_hdf5(H5Dread(dataset.get(), type, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, data),
                       "read restart dataset " + name);
        };
        read_dataset("pos_x", type_real, loaded.get_positions_x().data());
        read_dataset("pos_y", type_real, loaded.get_positions_y().data());
        read_dataset("pos_z", type_real, loaded.get_positions_z().data());
        read_dataset("mom_x", type_real, loaded.get_momenta_x().data());
        read_dataset("mom_y", type_real, loaded.get_momenta_y().data());
        read_dataset("mom_z", type_real, loaded.get_momenta_z().data());
        read_dataset("ParticleIDs", H5T_NATIVE_UINT64, loaded.get_ids().data());
        if (uniform_flag == 0) {
            read_dataset("masses", type_real, loaded.get_masses().data());
        }
    }

    loaded.set_verified_snapshot_ic_sha256(verified_snapshot_ic_sha256);
    require_valid_restart_payload(loaded);
    loaded.set_acceleration_validity(core::FieldValidity::INVALID);
    loaded.set_ghost_validity(core::FieldValidity::INVALID);

    time::TimeStepper candidate_stepper = stepper_out;
    candidate_stepper.restore_state(
        static_cast<std::size_t>(step),
        static_cast<core::Real>(current_a_disk));

    RestartStateIdentityResult identity{
        restart_state_sha256(
            config_, loaded, candidate_stepper,
            stored_dynamics_identity),
        true};
    if (identity.sha256 != stored_state_sha256) {
        throw std::runtime_error(
            "Restart state identity does not match decoded payload");
    }
    if (!expected_state_sha256.empty()
        && identity.sha256 != expected_state_sha256) {
        throw std::runtime_error(
            "Restart state identity does not match committed manifest");
    }

    // Commit caller-visible and run-local state only after every dynamics,
    // dataset, payload, state-identity, and step-boundary check has succeeded.
    config_.set_verified_snapshot_ic_sha256(verified_snapshot_ic_sha256);
    particles_out = std::move(loaded);
    stepper_out = std::move(candidate_stepper);
    return identity;
}

} // namespace io
} // namespace cosmo_nbody
