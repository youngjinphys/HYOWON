#include "cosmo_nbody/io/snapshot_descriptor.hpp"

namespace cosmo_nbody {
namespace io {
namespace {

void select_snapshot_scalar_range(
    hid_t file_space,
    std::size_t offset,
    std::size_t count,
    const std::string& name) {
    const hsize_t start[1] = {static_cast<hsize_t>(offset)};
    const hsize_t extent[1] = {static_cast<hsize_t>(count)};
    check_hdf5(
        H5Sselect_hyperslab(
            file_space,
            H5S_SELECT_SET,
            start,
            nullptr,
            extent,
            nullptr),
        "select snapshot scalar range " + name);
}

void read_snapshot_vector_range(
    hid_t group,
    const std::string& name,
    hid_t type_real,
    std::size_t global_count,
    std::size_t offset,
    std::size_t count,
    core::ParticleStore& loaded,
    core::Real current_a,
    bool velocities,
    Sha256Accumulator* raw_value_identity) {
    if (count > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::overflow_error(
            "Snapshot range vector buffer size overflows size_t");
    }

    auto dataset = open_dataset(group, name);
    require_vector_dataset_shape(dataset.get(), name, global_count);
    require_string_attr_value(
        dataset.get(),
        schema::ATTR_UNIT,
        velocities
            ? schema::VAL_VELOCITY_DATASET_UNIT
            : schema::VAL_COORDINATE_DATASET_UNIT,
        "Snapshot dataset " + name);
    if (velocities) {
        const std::string convention = read_string_attr(
            dataset.get(),
            schema::ATTR_VELOCITY_CONVENTION,
            schema::VAL_VELOCITY_CONVENTION.size());
        if (convention != schema::VAL_VELOCITY_CONVENTION) {
            throw std::runtime_error(
                "Unsupported velocity convention in snapshot: " + convention);
        }
    }

    std::vector<core::Real> buffer(count * 3);
    auto file_space = H5SpaceHandle::checked(
        H5Dget_space(dataset.get()),
        "snapshot range vector file space " + name);
    select_vector_hyperslab(file_space.get(), offset, count);
    const hsize_t memory_dims[2] = {
        static_cast<hsize_t>(count), 3};
    auto memory_space = H5SpaceHandle::checked(
        H5Screate_simple(2, memory_dims, nullptr),
        "snapshot range vector memory space " + name);
    check_hdf5(
        H5Dread(
            dataset.get(),
            type_real,
            memory_space.get(),
            file_space.get(),
            H5P_DEFAULT,
            buffer.data()),
        "read snapshot vector range " + name);
    if (raw_value_identity != nullptr) {
        raw_value_identity->update_canonical_real(buffer);
    }

    for (std::size_t local = 0; local < count; ++local) {
        const core::Real value0 = buffer[3 * local + 0];
        const core::Real value1 = buffer[3 * local + 1];
        const core::Real value2 = buffer[3 * local + 2];
        if (!std::isfinite(value0)
            || !std::isfinite(value1)
            || !std::isfinite(value2)) {
            throw std::runtime_error(
                "Snapshot vector range contains non-finite values: " + name);
        }
        if (velocities) {
            const core::Real p0 = value0 * current_a;
            const core::Real p1 = value1 * current_a;
            const core::Real p2 = value2 * current_a;
            if (!std::isfinite(p0) || !std::isfinite(p1)
                || !std::isfinite(p2)) {
                throw std::runtime_error(
                    "Snapshot velocity-to-momentum conversion overflowed");
            }
            loaded.get_momenta_x()[local] = p0;
            loaded.get_momenta_y()[local] = p1;
            loaded.get_momenta_z()[local] = p2;
        } else {
            loaded.get_positions_x()[local] = value0;
            loaded.get_positions_y()[local] = value1;
            loaded.get_positions_z()[local] = value2;
        }
    }
}

} // namespace

void SnapshotIO::read_exact_initial_condition_range(
    const VerifiedSnapshotSource& source,
    const SnapshotDescriptor& descriptor,
    std::size_t offset,
    std::size_t count,
    core::ParticleStore& particles_out) const {
    read_exact_initial_condition_range_with_raw_velocity_identity(
        source, descriptor, offset, count, particles_out, nullptr);
}

void SnapshotIO::read_exact_initial_condition_range_with_raw_velocity_identity(
    const VerifiedSnapshotSource& source,
    const SnapshotDescriptor& descriptor,
    std::size_t offset,
    std::size_t count,
    core::ParticleStore& particles_out,
    Sha256Accumulator* raw_velocity_identity) const {
    const SnapshotReadContext context = make_snapshot_read_context(config_);
    require_snapshot_read_context(context);

    if (descriptor.native_snapshot_object_sha256.empty()
        || descriptor.native_snapshot_object_sha256 != source.sha256()) {
        throw std::runtime_error(
            "Snapshot range descriptor does not identify the admitted exact source");
    }
    if (descriptor.particle_count
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "Snapshot range global population does not fit size_t");
    }
    const std::size_t global_count =
        static_cast<std::size_t>(descriptor.particle_count);
    SnapshotPopulationAdmission::exact(context.population_limit).require(
        global_count);
    if (count == 0) {
        throw std::invalid_argument(
            "Exact snapshot range must contain at least one particle");
    }
    if (offset > global_count || count > global_count - offset) {
        throw std::out_of_range(
            "Exact snapshot range lies outside the admitted particle population");
    }
    if (!snapshot_metadata_nearly_equal(
            descriptor.box_size_Mpc_h, context.box_size_Mpc_h)
        || !snapshot_metadata_nearly_equal(
            descriptor.omega_m, context.omega_m)
        || !snapshot_metadata_nearly_equal(
            descriptor.omega_lambda, context.omega_lambda)
        || !snapshot_metadata_nearly_equal(
            descriptor.hubble_param, context.hubble_param)) {
        throw std::runtime_error(
            "Snapshot range descriptor disagrees with the admitted simulation context");
    }
    if (!std::isfinite(descriptor.scale_factor)
        || descriptor.scale_factor <= 0.0) {
        throw std::runtime_error(
            "Snapshot range descriptor has an invalid scale factor");
    }

    const hid_t file = source.hdf5_file_id();
    if (file < 0) {
        throw std::logic_error(
            "Snapshot range source no longer owns an open HDF5 object");
    }
    require_hard_link(
        file, schema::GROUP_PART_TYPE_1, "Native snapshot /PartType1");
    auto part = H5GroupHandle::checked(
        H5Gopen2(
            file,
            schema::GROUP_PART_TYPE_1.c_str(),
            H5P_DEFAULT),
        "PartType1 group for exact snapshot range");

    core::ParticleStore loaded;
    if (descriptor.uniform_mass) {
        if (!std::isfinite(descriptor.uniform_particle_mass)
            || descriptor.uniform_particle_mass <= 0.0) {
            throw std::runtime_error(
                "Exact snapshot range has an invalid uniform mass descriptor");
        }
        loaded.set_uniform_mass(descriptor.uniform_particle_mass);
    } else {
        loaded.set_uniform_mass(std::nullopt);
    }
    loaded.resize(count);

    const hid_t type_real = sizeof(core::Real) == 8
        ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    read_snapshot_vector_range(
        part.get(),
        schema::DATASET_COORDINATES,
        type_real,
        global_count,
        offset,
        count,
        loaded,
        descriptor.scale_factor,
        false,
        nullptr);
    read_snapshot_vector_range(
        part.get(),
        schema::DATASET_VELOCITIES,
        type_real,
        global_count,
        offset,
        count,
        loaded,
        descriptor.scale_factor,
        true,
        raw_velocity_identity);

    {
        auto ids = open_dataset(part.get(), schema::DATASET_PARTICLE_IDS);
        require_scalar_dataset_shape(
            ids.get(), schema::DATASET_PARTICLE_IDS, global_count);
        require_string_attr_value(
            ids.get(), schema::ATTR_UNIT,
            schema::VAL_PARTICLE_ID_DATASET_UNIT,
            "Snapshot dataset " + schema::DATASET_PARTICLE_IDS);
        auto file_space = H5SpaceHandle::checked(
            H5Dget_space(ids.get()),
            "snapshot range particle-ID file space");
        select_snapshot_scalar_range(
            file_space.get(), offset, count, schema::DATASET_PARTICLE_IDS);
        const hsize_t memory_dims[1] = {static_cast<hsize_t>(count)};
        auto memory_space = H5SpaceHandle::checked(
            H5Screate_simple(1, memory_dims, nullptr),
            "snapshot range particle-ID memory space");
        check_hdf5(
            H5Dread(
                ids.get(),
                H5T_NATIVE_UINT64,
                memory_space.get(),
                file_space.get(),
                H5P_DEFAULT,
                loaded.get_ids().data()),
            "read snapshot particle-ID range");
        require_unique_ids_bounded(
            loaded.get_ids(),
            "Snapshot range",
            false);
    }

    const htri_t mass_exists = H5Lexists(
        part.get(), schema::DATASET_MASSES.c_str(), H5P_DEFAULT);
    if (mass_exists < 0) {
        throw std::runtime_error(
            "Failed to query Masses dataset existence for snapshot range");
    }
    if (descriptor.uniform_mass && mass_exists > 0) {
        throw std::runtime_error(
            "Snapshot has ambiguous type-1 mass representations in MassTable and Masses");
    }
    if (!descriptor.uniform_mass) {
        if (mass_exists == 0) {
            throw std::runtime_error(
                "Non-uniform snapshot is missing required Masses dataset");
        }
        auto masses = open_dataset(part.get(), schema::DATASET_MASSES);
        require_scalar_dataset_shape(
            masses.get(), schema::DATASET_MASSES, global_count);
        require_string_attr_value(
            masses.get(), schema::ATTR_UNIT,
            schema::VAL_MASS_DATASET_UNIT,
            "Snapshot dataset " + schema::DATASET_MASSES);
        auto file_space = H5SpaceHandle::checked(
            H5Dget_space(masses.get()),
            "snapshot range mass file space");
        select_snapshot_scalar_range(
            file_space.get(), offset, count, schema::DATASET_MASSES);
        const hsize_t memory_dims[1] = {static_cast<hsize_t>(count)};
        auto memory_space = H5SpaceHandle::checked(
            H5Screate_simple(1, memory_dims, nullptr),
            "snapshot range mass memory space");
        check_hdf5(
            H5Dread(
                masses.get(),
                type_real,
                memory_space.get(),
                file_space.get(),
                H5P_DEFAULT,
                loaded.get_masses().data()),
            "read snapshot mass range");
        for (const core::Real mass : loaded.get_masses()) {
            if (!std::isfinite(mass) || mass <= 0.0) {
                throw std::runtime_error(
                    "Snapshot range particle masses must be finite and positive");
            }
        }
    }

    const core::Real box_size = context.box_size_Mpc_h;
    for (std::size_t index = 0; index < count; ++index) {
        if (loaded.get_positions_x()[index] < 0.0
            || loaded.get_positions_x()[index] >= box_size
            || loaded.get_positions_y()[index] < 0.0
            || loaded.get_positions_y()[index] >= box_size
            || loaded.get_positions_z()[index] < 0.0
            || loaded.get_positions_z()[index] >= box_size) {
            throw std::runtime_error(
                "Snapshot range coordinates lie outside the admitted periodic box");
        }
    }

    loaded.set_acceleration_validity(core::FieldValidity::INVALID);
    loaded.set_ghost_validity(core::FieldValidity::INVALID);
    loaded.set_verified_snapshot_ic_sha256(source.sha256());
    particles_out = std::move(loaded);
}

} // namespace io
} // namespace cosmo_nbody
