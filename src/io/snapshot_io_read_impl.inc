
SnapshotReadContext make_snapshot_read_context(
    const config::SimulationParameters& config) {
    const std::uint64_t population = config.num_particles();
    if (population > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "Configured snapshot population does not fit size_t");
    }
    return {
        config.get_box().L,
        config.get_cosmology().omega_m,
        config.get_cosmology().omega_lambda,
        config.get_cosmology().h,
        static_cast<std::size_t>(population),
    };
}

void require_snapshot_read_context(
    const SnapshotReadContext& context) {
    if (!std::isfinite(context.box_size_Mpc_h)
        || context.box_size_Mpc_h <= 0.0
        || !std::isfinite(context.omega_m)
        || !std::isfinite(context.omega_lambda)
        || !std::isfinite(context.hubble_param)
        || context.hubble_param <= 0.0
        || context.population_limit == 0) {
        throw std::invalid_argument(
            "Snapshot read context is incomplete or non-physical");
    }
}

void read_snapshot_from_open_file(
    const SnapshotReadContext& context,
    hid_t file,
    core::ParticleStore& particles_out,
    core::Real& current_a_out,
    SnapshotReadPolicy policy,
    const SnapshotPopulationAdmission& population_admission) {
    require_snapshot_read_context(context);
    if (file < 0) {
        throw std::invalid_argument(
            "Snapshot read requires an open HDF5 file object");
    }
    const hid_t type_real = sizeof(core::Real) == 8
        ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

    require_hard_link(file, schema::GROUP_HEADER, "Native snapshot /Header");
    auto header = H5GroupHandle::checked(
        H5Gopen2(
            file,
            schema::GROUP_HEADER.c_str(),
            H5P_DEFAULT),
        "Header group");
    double a = 1.0;
    double redshift = 0.0;
    double box_size = 0.0;
    double omega_m = 0.0;
    double omega_lambda = 0.0;
    double hubble = 0.0;
    double mass_table[6] = {0, 0, 0, 0, 0, 0};
    std::uint32_t npart[6] = {0, 0, 0, 0, 0, 0};
    std::uint32_t npart_total[6] = {0, 0, 0, 0, 0, 0};
    std::int32_t num_files = 0;
    read_attr(header.get(), schema::ATTR_TIME, H5T_NATIVE_DOUBLE, &a);
    read_attr(
        header.get(), schema::ATTR_REDSHIFT,
        H5T_NATIVE_DOUBLE, &redshift);
    read_attr(
        header.get(), schema::ATTR_BOX_SIZE,
        H5T_NATIVE_DOUBLE, &box_size);
    read_attr(
        header.get(), schema::ATTR_OMEGA0,
        H5T_NATIVE_DOUBLE, &omega_m);
    read_attr(
        header.get(), schema::ATTR_OMEGA_LAMBDA,
        H5T_NATIVE_DOUBLE, &omega_lambda);
    read_attr(
        header.get(), schema::ATTR_HUBBLE_PARAM,
        H5T_NATIVE_DOUBLE, &hubble);
    read_attr(
        header.get(), schema::ATTR_MASS_TABLE,
        H5T_NATIVE_DOUBLE, mass_table);
    read_attr(
        header.get(), schema::ATTR_NUM_PART_THIS_FILE,
        H5T_NATIVE_UINT32, npart);
    read_attr(
        header.get(), schema::ATTR_NUM_PART_TOTAL,
        H5T_NATIVE_UINT32, npart_total);
    read_attr(
        header.get(), schema::ATTR_NUM_FILES,
        H5T_NATIVE_INT32, &num_files);
    if (!std::isfinite(a) || a <= 0.0) {
        throw std::runtime_error(
            "Snapshot contains non-finite or non-positive scale factor");
    }
    for (const double mass : mass_table) {
        if (!std::isfinite(mass) || mass < 0.0) {
            throw std::runtime_error(
                "Snapshot MassTable values must be finite and non-negative");
        }
    }
    if (!std::isfinite(redshift)
        || !snapshot_metadata_nearly_equal(redshift, 1.0 / a - 1.0)) {
        throw std::runtime_error(
            "Snapshot Header.Redshift is inconsistent with Header.Time");
    }
    if (!snapshot_metadata_nearly_equal(
            box_size, context.box_size_Mpc_h)
        || !snapshot_metadata_nearly_equal(omega_m, context.omega_m)
        || !snapshot_metadata_nearly_equal(
            omega_lambda, context.omega_lambda)
        || !snapshot_metadata_nearly_equal(
            hubble, context.hubble_param)) {
        throw std::runtime_error(
            "Snapshot box or cosmology header disagrees with the admitted read context");
    }
    if (num_files != 1) {
        throw std::runtime_error(
            "Serial snapshot reader requires NumFilesPerSnapshot=1");
    }
    for (std::size_t type = 0; type < 6; ++type) {
        if (npart[type] != npart_total[type]) {
            throw std::runtime_error(
                "Single-file snapshot NumPart_ThisFile and NumPart_Total disagree");
        }
        if (type != 1 && npart[type] != 0) {
            throw std::runtime_error(
                "Native one-fluid snapshot contains a particle population outside the effective collisionless PartType1 cb slot");
        }
    }

    // Native semantic metadata applies to every read. Exact-IC admission
    // performs additional configuration checks outside this low-level format
    // distinction, so no path may weaken the file checks.
    const auto require_unit = [&](const std::string& name, double expected) {
        if (!attribute_exists(header.get(), name)) {
            throw std::runtime_error(
                "Native snapshot is missing required unit attribute " + name);
        }
        double actual = 0.0;
        read_attr(header.get(), name, H5T_NATIVE_DOUBLE, &actual);
        if (!snapshot_metadata_nearly_equal(actual, expected)) {
            throw std::runtime_error(
                "Snapshot unit attribute disagrees with native code units: " + name);
        }
    };
    require_unit(
        schema::ATTR_UNIT_LENGTH_IN_CM,
        cosmology::units::UnitLength_in_cm);
    require_unit(
        schema::ATTR_UNIT_MASS_IN_G,
        cosmology::units::UnitMass_in_g);
    require_unit(
        schema::ATTR_UNIT_VELOCITY_IN_CM_PER_S,
        cosmology::units::UnitVelocity_in_cm_s);

    if (!link_exists(file, schema::GROUP_CONFIG)) {
        throw std::runtime_error(
            "Native snapshot is missing the Config semantic metadata group");
    }
    {
        auto source_config = H5GroupHandle::checked(
            H5Gopen2(
                file, schema::GROUP_CONFIG.c_str(), H5P_DEFAULT),
            "native snapshot Config group");
        require_hard_link(
            file, schema::GROUP_CONFIG, "Native snapshot /Config");
        for (const auto& binding : schema::NATIVE_SNAPSHOT_SEMANTIC_BINDINGS) {
            require_string_attr_value(
                source_config.get(),
                std::string(binding.attribute),
                std::string(binding.value),
                "Native snapshot Config");
        }
        require_nonempty_string_attr(
            source_config.get(),
            schema::ATTR_PHYSICS_FINGERPRINT,
            "Native snapshot Config",
            MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES);
        require_nonempty_string_attr(
            source_config.get(),
            schema::ATTR_RUN_METADATA_JSON,
            "Native snapshot Config",
            MAXIMUM_PERSISTED_PROVENANCE_TEXT_BYTES);
    }
    const core::Real current_a = static_cast<core::Real>(a);
    const std::size_t n = static_cast<std::size_t>(npart[1]);

    if (policy == SnapshotReadPolicy::ExactInitialConditions) {
        SnapshotPopulationAdmission::exact(context.population_limit).require(n);
    } else {
        // Analysis-subset snapshots may intentionally contain a reduced
        // shadow population, but may never use their header to allocate beyond
        // the admitted simulation/descriptor population.
        SnapshotPopulationAdmission::maximum(context.population_limit).require(n);
    }

    // A caller may impose an even narrower explicit admission. Both policy and
    // caller bounds are checked before ParticleStore allocates phase space.
    population_admission.require(n);

    const bool uniform = mass_table[1] > 0.0;
    core::ParticleStore loaded;
    if (uniform) {
        loaded.set_uniform_mass(
            static_cast<core::Real>(mass_table[1]));
    } else {
        loaded.set_uniform_mass(std::nullopt);
    }
    loaded.resize(n);

    if (n == 0) {
        loaded.set_acceleration_validity(core::FieldValidity::INVALID);
        loaded.set_ghost_validity(core::FieldValidity::INVALID);
        particles_out = std::move(loaded);
        current_a_out = current_a;
        return;
    }

    require_hard_link(
        file, schema::GROUP_PART_TYPE_1, "Native snapshot /PartType1");
    auto part = H5GroupHandle::checked(
        H5Gopen2(
            file,
            schema::GROUP_PART_TYPE_1.c_str(),
            H5P_DEFAULT),
        "PartType1 group");
    // Batching is an internal read-performance choice, not snapshot physics or
    // analysis context. Keep it bounded without inventing OutputParams.
    const std::size_t batch = std::max<std::size_t>(
        1, std::min<std::size_t>(n, SNAPSHOT_READ_BATCH_PARTICLES));
    read_vector_dataset_batched(
        part.get(),
        schema::DATASET_COORDINATES,
        type_real,
        n,
        batch,
        loaded,
        current_a,
        false);
    read_vector_dataset_batched(
        part.get(),
        schema::DATASET_VELOCITIES,
        type_real,
        n,
        batch,
        loaded,
        current_a,
        true);

    {
        auto ids = open_dataset(part.get(), schema::DATASET_PARTICLE_IDS);
        require_scalar_dataset_shape(
            ids.get(), schema::DATASET_PARTICLE_IDS, n);
        require_string_attr_value(
            ids.get(), schema::ATTR_UNIT,
            schema::VAL_PARTICLE_ID_DATASET_UNIT,
            "Snapshot dataset " + schema::DATASET_PARTICLE_IDS);
        check_hdf5(
            H5Dread(
                ids.get(),
                H5T_NATIVE_UINT64,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                loaded.get_ids().data()),
            "read particle IDs");
        require_unique_ids_bounded(
            loaded.get_ids(),
            "Snapshot",
            false);
    }

    const htri_t mass_exists = H5Lexists(
        part.get(), schema::DATASET_MASSES.c_str(), H5P_DEFAULT);
    if (mass_exists < 0) {
        throw std::runtime_error(
            "Failed to query Masses dataset existence");
    }
    if (uniform && mass_exists > 0) {
        throw std::runtime_error(
            "Snapshot has ambiguous type-1 mass representations in MassTable and Masses");
    }
    if (!uniform) {
        if (mass_exists == 0) {
            throw std::runtime_error(
                "Non-uniform snapshot is missing required Masses dataset");
        }
        auto masses = open_dataset(part.get(), schema::DATASET_MASSES);
        require_scalar_dataset_shape(
            masses.get(), schema::DATASET_MASSES, n);
        require_string_attr_value(
            masses.get(), schema::ATTR_UNIT,
            schema::VAL_MASS_DATASET_UNIT,
            "Snapshot dataset " + schema::DATASET_MASSES);
        check_hdf5(
            H5Dread(
                masses.get(),
                type_real,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                loaded.get_masses().data()),
            "read particle masses");
        for (const core::Real mass : loaded.get_masses()) {
            if (!std::isfinite(mass) || mass <= 0.0) {
                throw std::runtime_error(
                    "Snapshot particle masses must be finite and positive");
            }
        }
    }

    const core::Real L = context.box_size_Mpc_h;
    for (std::size_t index = 0; index < n; ++index) {
        if (loaded.get_positions_x()[index] < 0.0
            || loaded.get_positions_x()[index] >= L
            || loaded.get_positions_y()[index] < 0.0
            || loaded.get_positions_y()[index] >= L
            || loaded.get_positions_z()[index] < 0.0
            || loaded.get_positions_z()[index] >= L) {
            throw std::runtime_error(
                "Snapshot coordinates lie outside the admitted periodic box");
        }
    }

    loaded.set_acceleration_validity(core::FieldValidity::INVALID);
    loaded.set_ghost_validity(core::FieldValidity::INVALID);
    particles_out = std::move(loaded);
    current_a_out = current_a;
}

} // namespace

void read_snapshot_with_context(
    const SnapshotReadContext& context,
    const std::string& filename,
    core::ParticleStore& particles_out,
    core::Real& current_a_out,
    SnapshotReadPolicy policy,
    const SnapshotPopulationAdmission& population_admission) {
    auto file = H5FileHandle::checked(
        H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
        "open snapshot " + filename);
    read_snapshot_from_open_file(
        context,
        file.get(),
        particles_out,
        current_a_out,
        policy,
        population_admission);
}

void SnapshotIO::read_snapshot(
    const std::string& filename,
    core::ParticleStore& particles_out,
    core::Real& current_a_out,
    SnapshotReadPolicy policy,
    const SnapshotPopulationAdmission& population_admission) const {
    read_snapshot_with_context(
        make_snapshot_read_context(config_),
        filename,
        particles_out,
        current_a_out,
        policy,
        population_admission);
}

void SnapshotIO::read_snapshot(
    const VerifiedSnapshotSource& source,
    core::ParticleStore& particles_out,
    core::Real& current_a_out,
    SnapshotReadPolicy policy,
    const SnapshotPopulationAdmission& population_admission) const {
    read_snapshot_from_open_file(
        make_snapshot_read_context(config_),
        source.hdf5_file_id(),
        particles_out,
        current_a_out,
        policy,
        population_admission);
}

} // namespace io
} // namespace cosmo_nbody
