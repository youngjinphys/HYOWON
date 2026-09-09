} // namespace

SnapshotIO::SnapshotIO(const config::SimulationParameters& config)
    : config_(config) {}

void SnapshotIO::write_snapshot_payload_to_path(
    const core::ParticleStore& particles,
    core::Real current_a,
    const std::string& filename) const {
    if (filename.empty()) {
        throw std::invalid_argument(
            "Snapshot output path must not be empty");
    }
    require_finite_particle_payload(
        particles, current_a, config_.get_box().L);

    const std::size_t n = particles.num_owned_particles();
    if (n > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
            "Single-file snapshot header cannot represent particle count above uint32 range");
    }
    if (config_.get_box().N > std::numeric_limits<std::uint32_t>::max()
        || config_.get_box().N_mesh > std::numeric_limits<std::uint32_t>::max()
        || config_.ic_mesh_per_dimension()
            > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "Snapshot parameter metadata cannot represent N, N_mesh, or N_ic above uint32 range");
    }
    // `filename` is already the unique, unpublished staging path owned by
    // DurableFilePublication. HDF5 writes directly into that stage. The outer
    // publication transaction observes HDF5 close failures, fsyncs the closed
    // file, applies its explicit replace-or-require-absent publication policy,
    // and fsyncs the parent directory. The staging object itself must not
    // replace any pre-existing object; final-path replacement, when explicitly
    // requested by a caller, is solely the publication transaction's job.
    auto file = H5FileHandle::checked(
        H5Fcreate(
            filename.c_str(),
            H5F_ACC_EXCL,
            H5P_DEFAULT,
            H5P_DEFAULT),
        "create snapshot " + filename);
    const hid_t type_real = sizeof(core::Real) == 8
        ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

    auto header = H5GroupHandle::checked(
        H5Gcreate2(
            file.get(),
            schema::GROUP_HEADER.c_str(),
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        "Header group");

    std::uint32_t npart[6] = {
        0, static_cast<std::uint32_t>(n), 0, 0, 0, 0};
    double mass_table[6] = {0, 0, 0, 0, 0, 0};
    if (particles.get_uniform_mass().has_value()) {
        mass_table[1] = static_cast<double>(
            *particles.get_uniform_mass());
    }
    const double a = static_cast<double>(current_a);
    const double redshift = 1.0 / a - 1.0;
    const double box = static_cast<double>(config_.get_box().L);
    const std::int32_t one = 1;
    const std::int32_t dp_flag = sizeof(core::Real) == 8 ? 1 : 0;
    const double unit_length = cosmology::units::UnitLength_in_cm;
    const double unit_mass = cosmology::units::UnitMass_in_g;
    const double unit_velocity = cosmology::units::UnitVelocity_in_cm_s;

    write_attr_arr(
        header.get(), schema::ATTR_NUM_PART_THIS_FILE,
        H5T_NATIVE_UINT32, npart, 6);
    write_attr_arr(
        header.get(), schema::ATTR_NUM_PART_TOTAL,
        H5T_NATIVE_UINT32, npart, 6);
    write_attr_arr(
        header.get(), schema::ATTR_MASS_TABLE,
        H5T_NATIVE_DOUBLE, mass_table, 6);
    write_attr(header.get(), schema::ATTR_TIME, H5T_NATIVE_DOUBLE, &a);
    write_attr(
        header.get(), schema::ATTR_REDSHIFT,
        H5T_NATIVE_DOUBLE, &redshift);
    write_attr(
        header.get(), schema::ATTR_BOX_SIZE,
        H5T_NATIVE_DOUBLE, &box);

    const double om = config_.get_cosmology().omega_m;
    const double ol = config_.get_cosmology().omega_lambda;
    const double h = config_.get_cosmology().h;
    write_attr(header.get(), schema::ATTR_OMEGA0, H5T_NATIVE_DOUBLE, &om);
    write_attr(
        header.get(), schema::ATTR_OMEGA_LAMBDA,
        H5T_NATIVE_DOUBLE, &ol);
    write_attr(
        header.get(), schema::ATTR_HUBBLE_PARAM,
        H5T_NATIVE_DOUBLE, &h);
    write_attr(
        header.get(), schema::ATTR_NUM_FILES,
        H5T_NATIVE_INT32, &one);
    write_attr(
        header.get(), schema::ATTR_FLAG_DOUBLE_PRECISION,
        H5T_NATIVE_INT32, &dp_flag);
    write_attr(
        header.get(), schema::ATTR_UNIT_LENGTH_IN_CM,
        H5T_NATIVE_DOUBLE, &unit_length);
    write_attr(
        header.get(), schema::ATTR_UNIT_MASS_IN_G,
        H5T_NATIVE_DOUBLE, &unit_mass);
    write_attr(
        header.get(), schema::ATTR_UNIT_VELOCITY_IN_CM_PER_S,
        H5T_NATIVE_DOUBLE, &unit_velocity);

    const RunMetadata run_metadata = RunMetadata::from_config(
        config_, ProductLineage::DirectSimulation, {}, true);
    const std::uint64_t metadata_dynamic_payload_bytes =
        run_metadata_dynamic_payload_bytes(run_metadata);
    const std::string metadata_json = run_metadata.to_json();
    const std::string physics_fingerprint =
        snapshot_physics_fingerprint(config_);
    (void)snapshot_metadata_memory_plan(
        metadata_dynamic_payload_bytes,
        metadata_json.size(),
        physics_fingerprint.size());

    auto params = H5GroupHandle::checked(
        H5Gcreate2(
            file.get(),
            schema::GROUP_PARAMETERS.c_str(),
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        "Parameters group");
    const std::uint64_t seed = run_metadata.seed;
    const std::uint32_t npart_dim =
        static_cast<std::uint32_t>(config_.get_box().N);
    const std::uint32_t nmesh =
        static_cast<std::uint32_t>(config_.get_box().N_mesh);
    const std::uint32_t ic_mesh = static_cast<std::uint32_t>(
        run_metadata.ic_mesh_per_dimension);
    const double eps = static_cast<double>(config_.eps());
    const double sigma8 = config_.get_cosmology().sigma8;
    const double ns = config_.get_cosmology().n_s;
    const double omega_b = config_.get_cosmology().omega_b;
    const std::int32_t lpt_order = run_metadata.lpt_order;
    write_attr(params.get(), schema::ATTR_SEED, H5T_NATIVE_UINT64, &seed);
    write_attr(
        params.get(), schema::ATTR_PARTICLES_PER_DIM,
        H5T_NATIVE_UINT32, &npart_dim);
    write_attr(
        params.get(), schema::ATTR_PM_MESH_PER_DIM,
        H5T_NATIVE_UINT32, &nmesh);
    write_attr(
        params.get(), schema::ATTR_IC_MESH_PER_DIM,
        H5T_NATIVE_UINT32, &ic_mesh);
    write_string_attr(
        params.get(), schema::ATTR_IC_AMPLITUDE_MODE,
        run_metadata.ic_amplitude_mode);
    write_string_attr(
        params.get(), schema::ATTR_IC_PHASE_PAIRING,
        run_metadata.ic_phase_pairing);
    write_attr(
        params.get(), schema::ATTR_SOFTENING_COMOVING,
        H5T_NATIVE_DOUBLE, &eps);
    write_attr(params.get(), schema::ATTR_SIGMA8, H5T_NATIVE_DOUBLE, &sigma8);
    write_attr(params.get(), schema::ATTR_N_S, H5T_NATIVE_DOUBLE, &ns);
    write_attr(params.get(), schema::ATTR_OMEGA_B, H5T_NATIVE_DOUBLE, &omega_b);
    write_attr(
        params.get(), schema::ATTR_LPT_ORDER,
        H5T_NATIVE_INT32, &lpt_order);
    write_string_attr(
        params.get(), schema::ATTR_POWER_SPECTRUM_FIDELITY,
        run_metadata.power_spectrum_fidelity);

    {
        auto config_creation = provenance_group_creation_properties(
            "Config group creation");
        auto config_group = H5GroupHandle::checked(
            H5Gcreate2(
                file.get(), schema::GROUP_CONFIG.c_str(), H5P_DEFAULT,
                config_creation.get(), H5P_DEFAULT),
            "Config group");
        write_string_attr(
            config_group.get(), schema::ATTR_SOFTWARE_NAME,
            std::string(schema::VAL_SOFTWARE_NAME));
        write_string_attr(
            config_group.get(), schema::ATTR_SNAPSHOT_SCHEMA,
            std::string(schema::VAL_SNAPSHOT_SCHEMA));
        write_string_attr(
            config_group.get(), schema::ATTR_PHYSICS_FINGERPRINT,
            physics_fingerprint);
        write_string_attr(
            config_group.get(), schema::ATTR_RUN_METADATA_JSON,
            metadata_json);
        for (const auto& binding : schema::NATIVE_SNAPSHOT_SEMANTIC_BINDINGS) {
            write_string_attr(
                config_group.get(),
                std::string(binding.attribute),
                std::string(binding.value));
        }
    }

    if (n != 0) {
        auto part = H5GroupHandle::checked(
            H5Gcreate2(
                file.get(),
                schema::GROUP_PART_TYPE_1.c_str(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT),
            "PartType1 group");
        const std::size_t batch = checked_batch_particles(config_, n);
        write_vector_dataset_batched(
            part.get(),
            schema::DATASET_COORDINATES,
            type_real,
            n,
            batch,
            particles,
            current_a,
            false);
        write_vector_dataset_batched(
            part.get(),
            schema::DATASET_VELOCITIES,
            type_real,
            n,
            batch,
            particles,
            current_a,
            true);

        const hsize_t dims1[1] = {static_cast<hsize_t>(n)};
        auto space1 = H5SpaceHandle::checked(
            H5Screate_simple(1, dims1, nullptr),
            "particle scalar space");
        {
            auto dataset = create_dataset(
                part.get(),
                schema::DATASET_PARTICLE_IDS,
                H5T_NATIVE_UINT64,
                space1.get());
            write_string_attr(
                dataset.get(), schema::ATTR_UNIT,
                schema::VAL_PARTICLE_ID_DATASET_UNIT);
            check_hdf5(
                H5Dwrite(
                    dataset.get(),
                    H5T_NATIVE_UINT64,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    particles.get_ids().data()),
                "write particle IDs");
        }

        if (!particles.get_uniform_mass().has_value()) {
            auto dataset = create_dataset(
                part.get(),
                schema::DATASET_MASSES,
                type_real,
                space1.get());
            write_string_attr(
                dataset.get(), schema::ATTR_UNIT,
                schema::VAL_MASS_DATASET_UNIT);
            check_hdf5(
                H5Dwrite(
                    dataset.get(),
                    type_real,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    particles.get_masses().data()),
                "write particle masses");
        }
    }

    // Catch an HDF5 flush error while the file handle is still available for
    // a contextual diagnostic. The subsequent RAII close is independently
    // observed by DurableFilePublication before final-path publication.
    check_hdf5(
        H5Fflush(file.get(), H5F_SCOPE_GLOBAL),
        "flush snapshot file");
}

namespace {
