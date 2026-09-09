    SnapshotPayloadIdentity local_expected_payload_identity{};
    std::exception_ptr payload_identity_exception;
    try {
        local_expected_payload_identity =
            compute_expected_snapshot_payload_identity(
                local_particles,
                current_a,
                transfer_batch,
                write_explicit_masses,
                batch_buffers);
    } catch (...) {
        payload_identity_exception = std::current_exception();
    }
    synchronize_distributed_snapshot_exception(
        payload_identity_exception, mpi_size,
        "Distributed snapshot source-payload identity construction");

    std::vector<char> expected_payload_identities;
    payload_identity_exception = nullptr;
    try {
        expected_payload_identities.resize(
            static_cast<std::size_t>(mpi_size)
            * snapshot_payload_identity_bytes);
    } catch (...) {
        payload_identity_exception = std::current_exception();
    }
    synchronize_distributed_snapshot_exception(
        payload_identity_exception, mpi_size,
        "Distributed snapshot payload-identity table allocation");
    require_mpi_success(
        MPI_Allgather(
            local_expected_payload_identity.data(),
            static_cast<int>(local_expected_payload_identity.size()),
            MPI_CHAR,
            expected_payload_identities.data(),
            static_cast<int>(local_expected_payload_identity.size()),
            MPI_CHAR,
            MPI_COMM_WORLD),
        "MPI_Allgather for distributed snapshot source-payload identities");

    constexpr int tag_batch_header = 7101;
    constexpr int tag_coordinates = 7102;
    constexpr int tag_velocities = 7103;
    constexpr int tag_ids = 7104;
    constexpr int tag_masses = 7105;

    std::unique_ptr<DurableFilePublication> publication;
    H5FileHandle file;
    H5DatasetHandle coordinates;
    H5DatasetHandle velocities;
    H5DatasetHandle ids;
    H5DatasetHandle masses;
    const hid_t type_real = sizeof(core::Real) == 8
        ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    const std::uint64_t hdf5_close_failure_epoch_before_write =
        current_hdf5_close_failure_epoch();

    // Filesystem and HDF5 setup is deliberately rank-0-only. No non-root rank
    // resolves, stats, opens, locks, links, or fsyncs the output path.
    std::exception_ptr setup_exception;
    if (rank == 0) {
        try {
            std::ostringstream filename_builder;
            filename_builder
                << "snapshot_" << snapshot_index << ".hdf5";
            const std::filesystem::path output_directory =
                std::filesystem::absolute(output_directory_)
                    .lexically_normal();
            const std::filesystem::path final_path =
                (output_directory / filename_builder.str())
                    .lexically_normal();
            publication = std::make_unique<DurableFilePublication>(
                final_path,
                "distributed native snapshot publication",
                DurableFilePublicationPolicy::RequireAbsent);

            file = H5FileHandle::checked(
                H5Fcreate(
                    publication->staging_path().string().c_str(),
                    H5F_ACC_EXCL,
                    H5P_DEFAULT,
                    H5P_DEFAULT),
                "create distributed snapshot staging file");

            {
                auto header = H5GroupHandle::checked(
                    H5Gcreate2(
                        file.get(), schema::GROUP_HEADER.c_str(),
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "distributed snapshot Header group");

                std::uint32_t npart[6] = {
                    0, static_cast<std::uint32_t>(global_count),
                    0, 0, 0, 0};
                double mass_table[6] = {0, 0, 0, 0, 0, 0};
                if (mass_representation.uniform) {
                    mass_table[1] = mass_representation.uniform_mass;
                }
                const double a = static_cast<double>(current_a);
                const double redshift = 1.0 / a - 1.0;
                const double box = static_cast<double>(config_.get_box().L);
                const double omega_m = config_.get_cosmology().omega_m;
                const double omega_lambda =
                    config_.get_cosmology().omega_lambda;
                const double hubble = config_.get_cosmology().h;
                const std::int32_t one = 1;
                const std::int32_t dp_flag = sizeof(core::Real) == 8 ? 1 : 0;
                const double unit_length =
                    cosmology::units::UnitLength_in_cm;
                const double unit_mass =
                    cosmology::units::UnitMass_in_g;
                const double unit_velocity =
                    cosmology::units::UnitVelocity_in_cm_s;

                write_attr_array(
                    header.get(), schema::ATTR_NUM_PART_THIS_FILE,
                    H5T_NATIVE_UINT32, npart, 6);
                write_attr_array(
                    header.get(), schema::ATTR_NUM_PART_TOTAL,
                    H5T_NATIVE_UINT32, npart, 6);
                write_attr_array(
                    header.get(), schema::ATTR_MASS_TABLE,
                    H5T_NATIVE_DOUBLE, mass_table, 6);
                write_attr(
                    header.get(), schema::ATTR_TIME,
                    H5T_NATIVE_DOUBLE, &a);
                write_attr(
                    header.get(), schema::ATTR_REDSHIFT,
                    H5T_NATIVE_DOUBLE, &redshift);
                write_attr(
                    header.get(), schema::ATTR_BOX_SIZE,
                    H5T_NATIVE_DOUBLE, &box);
                write_attr(
                    header.get(), schema::ATTR_OMEGA0,
                    H5T_NATIVE_DOUBLE, &omega_m);
                write_attr(
                    header.get(), schema::ATTR_OMEGA_LAMBDA,
                    H5T_NATIVE_DOUBLE, &omega_lambda);
                write_attr(
                    header.get(), schema::ATTR_HUBBLE_PARAM,
                    H5T_NATIVE_DOUBLE, &hubble);
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
                    header.get(),
                    schema::ATTR_UNIT_VELOCITY_IN_CM_PER_S,
                    H5T_NATIVE_DOUBLE, &unit_velocity);
            }

            {
                auto params = H5GroupHandle::checked(
                    H5Gcreate2(
                        file.get(), schema::GROUP_PARAMETERS.c_str(),
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "distributed snapshot Parameters group");
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
                write_attr(
                    params.get(), schema::ATTR_SEED,
                    H5T_NATIVE_UINT64, &seed);
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
                write_attr(
                    params.get(), schema::ATTR_SIGMA8,
                    H5T_NATIVE_DOUBLE, &sigma8);
                write_attr(
                    params.get(), schema::ATTR_N_S,
                    H5T_NATIVE_DOUBLE, &ns);
                write_attr(
                    params.get(), schema::ATTR_OMEGA_B,
                    H5T_NATIVE_DOUBLE, &omega_b);
                write_attr(
                    params.get(), schema::ATTR_LPT_ORDER,
                    H5T_NATIVE_INT32, &lpt_order);
                write_string_attr(
                    params.get(), schema::ATTR_POWER_SPECTRUM_FIDELITY,
                    run_metadata.power_spectrum_fidelity);
            }

            {
                auto config_creation = provenance_group_creation_properties(
                    "distributed snapshot Config group creation");
                auto config_group = H5GroupHandle::checked(
                    H5Gcreate2(
                        file.get(), schema::GROUP_CONFIG.c_str(),
                        H5P_DEFAULT, config_creation.get(), H5P_DEFAULT),
                    "distributed snapshot Config group");
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
                for (const auto& binding
                     : schema::NATIVE_SNAPSHOT_SEMANTIC_BINDINGS) {
                    write_string_attr(
                        config_group.get(),
                        std::string(binding.attribute),
                        std::string(binding.value));
                }
            }

            if (global_count != 0) {
                auto particles_group = H5GroupHandle::checked(
                    H5Gcreate2(
                        file.get(), schema::GROUP_PART_TYPE_1.c_str(),
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "distributed snapshot PartType1 group");
                const hsize_t global_vector_dims[2] = {
                    static_cast<hsize_t>(global_count), 3};
                const hsize_t global_scalar_dims[1] = {
                    static_cast<hsize_t>(global_count)};
                auto vector_space = H5SpaceHandle::checked(
                    H5Screate_simple(
                        2, global_vector_dims, nullptr),
                    "distributed snapshot global vector space");
                auto scalar_space = H5SpaceHandle::checked(
                    H5Screate_simple(
                        1, global_scalar_dims, nullptr),
                    "distributed snapshot global scalar space");

                coordinates = H5DatasetHandle::checked(
                    H5Dcreate2(
                        particles_group.get(),
                        schema::DATASET_COORDINATES.c_str(),
                        detail::canonical_snapshot_file_type(
                            type_real,
                            "distributed snapshot Coordinates dataset"),
                        vector_space.get(),
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "distributed snapshot Coordinates dataset");
                write_string_attr(
                    coordinates.get(), schema::ATTR_UNIT,
                    schema::VAL_COORDINATE_DATASET_UNIT);
                velocities = H5DatasetHandle::checked(
                    H5Dcreate2(
                        particles_group.get(),
                        schema::DATASET_VELOCITIES.c_str(),
                        detail::canonical_snapshot_file_type(
                            type_real,
                            "distributed snapshot Velocities dataset"),
                        vector_space.get(),
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "distributed snapshot Velocities dataset");
                write_string_attr(
                    velocities.get(), schema::ATTR_VELOCITY_CONVENTION,
                    schema::VAL_VELOCITY_CONVENTION);
                write_string_attr(
                    velocities.get(), schema::ATTR_UNIT,
                    schema::VAL_VELOCITY_DATASET_UNIT);
                ids = H5DatasetHandle::checked(
                    H5Dcreate2(
                        particles_group.get(),
                        schema::DATASET_PARTICLE_IDS.c_str(),
                        detail::canonical_snapshot_file_type(
                            H5T_NATIVE_UINT64,
                            "distributed snapshot ParticleIDs dataset"),
                        scalar_space.get(),
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    "distributed snapshot ParticleIDs dataset");
                write_string_attr(
                    ids.get(), schema::ATTR_UNIT,
                    schema::VAL_PARTICLE_ID_DATASET_UNIT);
                if (write_explicit_masses) {
                    masses = H5DatasetHandle::checked(
                        H5Dcreate2(
                            particles_group.get(),
                            schema::DATASET_MASSES.c_str(),
                            detail::canonical_snapshot_file_type(
                                type_real,
                                "distributed snapshot Masses dataset"),
                            scalar_space.get(),
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                        "distributed snapshot Masses dataset");
                    write_string_attr(
                        masses.get(), schema::ATTR_UNIT,
                        schema::VAL_MASS_DATASET_UNIT);
                }
            }
            require_no_hdf5_close_failures_since(
                hdf5_close_failure_epoch_before_write,
                "distributed snapshot schema setup");
        } catch (...) {
            setup_exception = std::current_exception();
        }
    }
    if (rank == 0 && setup_exception) {
        std::exception_ptr close_exception;
        masses.reset();
        ids.reset();
        velocities.reset();
        coordinates.reset();
        file.reset();
        try {
            require_no_hdf5_close_failures_since(
                hdf5_close_failure_epoch_before_write,
                "failed distributed snapshot schema setup");
        } catch (...) {
            close_exception = std::current_exception();
        }
        setup_exception = combine_snapshot_exceptions(
            setup_exception,
            close_exception,
            "Distributed snapshot schema setup/close");

        if (publication && !publication->canonical_visible()) {
            try {
                publication->discard_unpublished();
            } catch (...) {
                setup_exception = combine_snapshot_exceptions(
                    setup_exception,
                    std::current_exception(),
                    "Distributed snapshot setup-stage cleanup");
            }
        }
    }
    synchronize_distributed_snapshot_exception(
        setup_exception, mpi_size,
        "Distributed snapshot rank-0 file setup");

    const auto write_root_batch = [&] (
        std::uint64_t batch_offset,
        std::size_t count) {
        {
            auto file_space = H5SpaceHandle::checked(
                H5Dget_space(coordinates.get()),
                "distributed Coordinates file space");
            select_file_hyperslab_2d(
                file_space.get(), batch_offset, count);
            auto memory_space = make_memory_space_2d(count);
            check_hdf5(
                H5Dwrite(
                    coordinates.get(), type_real,
                    memory_space.get(), file_space.get(), H5P_DEFAULT,
                    batch_buffers.coordinates.data()),
                "write distributed Coordinates batch");
        }
        {
            auto file_space = H5SpaceHandle::checked(
                H5Dget_space(velocities.get()),
                "distributed Velocities file space");
            select_file_hyperslab_2d(
                file_space.get(), batch_offset, count);
            auto memory_space = make_memory_space_2d(count);
            check_hdf5(
                H5Dwrite(
                    velocities.get(), type_real,
                    memory_space.get(), file_space.get(), H5P_DEFAULT,
                    batch_buffers.velocities.data()),
                "write distributed Velocities batch");
        }
        {
            auto file_space = H5SpaceHandle::checked(
                H5Dget_space(ids.get()),
                "distributed ParticleIDs file space");
            select_file_hyperslab_1d(
                file_space.get(), batch_offset, count);
            auto memory_space = make_memory_space_1d(count);
            check_hdf5(
                H5Dwrite(
                    ids.get(), H5T_NATIVE_UINT64,
                    memory_space.get(), file_space.get(), H5P_DEFAULT,
                    batch_buffers.ids.data()),
                "write distributed ParticleIDs batch");
        }
        if (write_explicit_masses) {
            auto file_space = H5SpaceHandle::checked(
                H5Dget_space(masses.get()),
                "distributed Masses file space");
            select_file_hyperslab_1d(
                file_space.get(), batch_offset, count);
            auto memory_space = make_memory_space_1d(count);
            check_hdf5(
                H5Dwrite(
                    masses.get(), type_real,
                    memory_space.get(), file_space.get(), H5P_DEFAULT,
                    batch_buffers.masses.data()),
                "write distributed Masses batch");
        }
    };

    std::exception_ptr write_exception;
    bool batch_announcement_mismatch = false;
    const auto preserve_first_write_exception = [&] (
        std::exception_ptr candidate) {
        if (!write_exception) write_exception = std::move(candidate);
    };
    const core::Real inv_a = 1.0 / current_a;
    const MPI_Datatype real_datatype = mpi_real_datatype();

    // Ranks are serialized in their exact global-offset order. This bounds
    // rank-0 memory by one configured batch and makes every HDF5 write serial.
    // Crucially, rank 0 receives every component of a remote batch before it
    // attempts HDF5. Once an HDF5 error is recorded, it keeps draining all
    // later batches and skips only the failed storage operations.
    for (int source = 0; source < mpi_size; ++source) {
        const std::size_t source_index = static_cast<std::size_t>(source);
        const std::uint64_t source_count = rank_counts[source_index];
        const std::uint64_t source_offset = rank_offsets[source_index];
        for (std::uint64_t begin64 = 0; begin64 < source_count;
             begin64 += transfer_batch) {
            const std::uint64_t count64 = std::min(
                transfer_batch, source_count - begin64);
            const std::size_t begin = static_cast<std::size_t>(begin64);
            const std::size_t count = static_cast<std::size_t>(count64);
            const int scalar_count = static_cast<int>(count);
            const int vector_count = static_cast<int>(count * 3);
            const std::uint64_t batch_offset = source_offset + begin64;

            if (rank == source) {
                pack_batch(
                    local_particles,
                    inv_a,
                    begin,
                    count,
                    write_explicit_masses,
                    batch_buffers);
            }

            if (source == 0) {
                if (rank == 0) {
                    if (!write_exception) {
                        try {
                            write_root_batch(batch_offset, count);
                        } catch (...) {
                            preserve_first_write_exception(
                                std::current_exception());
                        }
                    }
                }
                continue;
            }

            if (rank == source) {
                const std::uint64_t header[2] = {
                    batch_offset, count64};
                require_mpi_success(
                    MPI_Send(
                        header, 2, MPI_UINT64_T, 0,
                        tag_batch_header, MPI_COMM_WORLD),
                    "MPI_Send for distributed snapshot batch header");
                require_mpi_success(
                    MPI_Send(
                        batch_buffers.coordinates.data(),
                        vector_count, real_datatype, 0,
                        tag_coordinates, MPI_COMM_WORLD),
                    "MPI_Send for distributed snapshot Coordinates");
                require_mpi_success(
                    MPI_Send(
                        batch_buffers.velocities.data(),
                        vector_count, real_datatype, 0,
                        tag_velocities, MPI_COMM_WORLD),
                    "MPI_Send for distributed snapshot Velocities");
                require_mpi_success(
                    MPI_Send(
                        batch_buffers.ids.data(),
                        scalar_count, MPI_UINT64_T, 0,
                        tag_ids, MPI_COMM_WORLD),
                    "MPI_Send for distributed snapshot ParticleIDs");
                if (write_explicit_masses) {
                    require_mpi_success(
                        MPI_Send(
                            batch_buffers.masses.data(),
                            scalar_count, real_datatype, 0,
                            tag_masses, MPI_COMM_WORLD),
                        "MPI_Send for distributed snapshot Masses");
                }
            } else if (rank == 0) {
                std::uint64_t announced[2] = {0, 0};
                require_mpi_success(
                    MPI_Recv(
                        announced, 2, MPI_UINT64_T, source,
                        tag_batch_header, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE),
                    "MPI_Recv for distributed snapshot batch header");
                require_mpi_success(
                    MPI_Recv(
                        batch_buffers.coordinates.data(),
                        vector_count, real_datatype, source,
                        tag_coordinates, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE),
                    "MPI_Recv for distributed snapshot Coordinates");
                require_mpi_success(
                    MPI_Recv(
                        batch_buffers.velocities.data(),
                        vector_count, real_datatype, source,
                        tag_velocities, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE),
                    "MPI_Recv for distributed snapshot Velocities");
                require_mpi_success(
                    MPI_Recv(
                        batch_buffers.ids.data(),
                        scalar_count, MPI_UINT64_T, source,
                        tag_ids, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE),
                    "MPI_Recv for distributed snapshot ParticleIDs");
                if (write_explicit_masses) {
                    require_mpi_success(
                        MPI_Recv(
                            batch_buffers.masses.data(),
                            scalar_count, real_datatype, source,
                            tag_masses, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE),
                        "MPI_Recv for distributed snapshot Masses");
                }

                if ((announced[0] != batch_offset
                     || announced[1] != count64)
                    && !write_exception) {
                    // Keep the live send/receive protocol allocation-free on
                    // this error path. Materialize a diagnostic only after all
                    // ranks have completed every bounded transfer.
                    batch_announcement_mismatch = true;
                }
                if (!write_exception) {
                    try {
                        write_root_batch(batch_offset, count);
                    } catch (...) {
                        preserve_first_write_exception(
                            std::current_exception());
                    }
                }
            }
        }
    }

    if (rank == 0 && batch_announcement_mismatch && !write_exception) {
        try {
            throw std::runtime_error(
                "Distributed snapshot batch announcement disagrees with the preflight rank interval");
        } catch (...) {
            write_exception = std::current_exception();
        }
    }

    // Transport is complete on every rank. Release its bounded staging buffers
    // before rank 0 allocates bounded native-reader validation ranges.
    batch_buffers = BatchBuffers{};

    if (rank == 0) {
        std::exception_ptr finalization_exception;
        try {
            check_hdf5(
                H5Fflush(file.get(), H5F_SCOPE_GLOBAL),
                "flush distributed snapshot staging file");
        } catch (...) {
            finalization_exception = std::current_exception();
        }

        masses.reset();
        ids.reset();
        velocities.reset();
        coordinates.reset();
        file.reset();
        try {
            require_no_hdf5_close_failures_since(
                hdf5_close_failure_epoch_before_write,
                "distributed snapshot staging write");
        } catch (...) {
            if (!finalization_exception) {
                finalization_exception = std::current_exception();
            }
        }
        preserve_first_write_exception(finalization_exception);

        if (!write_exception) {
            try {
                {
                    VerifiedSnapshotSource source(
                        publication->staging_path());
                    const SnapshotDescriptor descriptor =
                        validate_stage_metadata(
                            source,
                            config_,
                            current_a,
                            global_count,
                            mass_representation,
                            physics_fingerprint,
                            metadata_json,
                            run_metadata);
                    SnapshotIO native_reader(config_);
                    core::ParticleStore validation_batch;
                    const auto validation_capacity = static_cast<std::size_t>(std::min(
                        transfer_batch, *std::max_element(rank_counts.begin(), rank_counts.end())));
                    batch_buffers.coordinates.resize(
                        validation_capacity * 3U);
                    batch_buffers.velocities.resize(
                        validation_capacity * 3U);
                    for (int expected_rank = 0;
                         expected_rank < mpi_size; ++expected_rank) {
                        Sha256Accumulator observed_coordinates;
                        Sha256Accumulator observed_raw_velocities;
                        Sha256Accumulator observed_momenta;
                        Sha256Accumulator observed_ids;
                        Sha256Accumulator observed_masses;
                        const std::size_t expected_rank_index =
                            static_cast<std::size_t>(expected_rank);
                        const std::uint64_t rank_count =
                            rank_counts[expected_rank_index];
                        const std::uint64_t rank_offset =
                            rank_offsets[expected_rank_index];
                        for (std::uint64_t begin = 0; begin < rank_count;
                             begin += transfer_batch) {
                            const std::uint64_t validation_count64 = std::min(
                                transfer_batch, rank_count - begin);
                            const std::size_t validation_count =
                                static_cast<std::size_t>(validation_count64);
                            validation_batch = core::ParticleStore{};
                            native_reader
                                .read_exact_initial_condition_range_with_raw_velocity_identity(
                                source,
                                descriptor,
                                static_cast<std::size_t>(rank_offset + begin),
                                validation_count,
                                validation_batch,
                                &observed_raw_velocities);
                            if (validation_batch.num_owned_particles()
                                != validation_count) {
                                throw std::runtime_error(
                                    "Distributed snapshot bounded stage payload count mismatch");
                            }
                            update_observed_snapshot_payload_identity(
                                validation_batch,
                                write_explicit_masses,
                                batch_buffers,
                                observed_coordinates,
                                observed_momenta,
                                observed_ids,
                                observed_masses);
                        }
                        const SnapshotPayloadIdentity observed_identity =
                            finish_snapshot_payload_identity(
                                observed_coordinates,
                                observed_raw_velocities,
                                observed_momenta,
                                observed_ids,
                                observed_masses);
                        const auto expected_begin =
                            expected_payload_identities.begin()
                            + static_cast<std::ptrdiff_t>(
                                expected_rank_index
                                * snapshot_payload_identity_bytes);
                        if (!std::equal(
                                observed_identity.begin(),
                                observed_identity.end(),
                                expected_begin)) {
                            throw std::runtime_error(
                                "Distributed snapshot staged phase-space payload disagrees exactly with the source particle state");
                        }
                    }

                    // VerifiedSnapshotSource hashes the complete closed stage
                    // once before these reads. A second full-file hash is not
                    // needed here: the stage has a unique unpublished name,
                    // is owned by this DurableFilePublication, and no writer
                    // retains an HDF5 handle. Non-cooperating mutation of that
                    // private stage is outside the publication threat model.
                }
                require_no_hdf5_close_failures_since(
                    hdf5_close_failure_epoch_before_write,
                    "distributed snapshot native stage validation");

                publication->publish_nonempty();
            } catch (...) {
                preserve_first_write_exception(
                    std::current_exception());
            }
        }
    }

    if (rank == 0 && write_exception && publication
        && !publication->canonical_visible()) {
        const std::exception_ptr primary_exception = write_exception;
        try {
            publication->discard_unpublished();
        } catch (...) {
            const std::exception_ptr cleanup_exception =
                std::current_exception();
            try {
                write_exception = std::make_exception_ptr(
                    std::runtime_error(
                        "Distributed snapshot publication failed before visibility ("
                        + snapshot_exception_text(primary_exception)
                        + "); explicit unpublished-stage cleanup also failed ("
                        + snapshot_exception_text(cleanup_exception)
                        + "). The unpublished stage was retained for inspection."));
            } catch (...) {
                // Keep a reportable cleanup exception rather than escaping this
                // rank before the final collective error synchronization.
                write_exception = cleanup_exception;
            }
        }
    }

    synchronize_distributed_snapshot_exception(
        write_exception, mpi_size,
        "Distributed snapshot rank-0 serial HDF5 publication");
