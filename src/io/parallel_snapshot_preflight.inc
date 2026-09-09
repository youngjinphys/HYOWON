    runtime::require_active_mpi_main_thread(
        "ParallelSnapshotIO::write_snapshot_payload");

    int rank = 0;
    require_mpi_success(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank for parallel snapshot");
    int mpi_size = 0;
    require_mpi_success(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size),
        "MPI_Comm_size for parallel snapshot");
    if (mpi_size < 1 || rank < 0 || rank >= mpi_size) {
        throw std::runtime_error(
            "Distributed snapshot has an invalid MPI topology");
    }

    const std::size_t local_owned = local_particles.num_owned_particles();
    int local_metadata_invalid = 0;
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (local_owned > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            local_metadata_invalid = 1;
        }
    }
    if (!std::isfinite(current_a) || current_a <= 0.0
        || config_.get_box().N > std::numeric_limits<std::uint32_t>::max()
        || config_.get_box().N_mesh > std::numeric_limits<std::uint32_t>::max()
        || config_.ic_mesh_per_dimension()
            > std::numeric_limits<std::uint32_t>::max()) {
        local_metadata_invalid = 1;
    }
    int any_metadata_invalid = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_metadata_invalid, &any_metadata_invalid,
            1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for parallel snapshot metadata validation");
    if (any_metadata_invalid != 0) {
        throw std::invalid_argument(
            "Parallel snapshot metadata is invalid on at least one rank");
    }

    const double local_a = static_cast<double>(current_a);
    double a_min = 0.0;
    double a_max = 0.0;
    require_mpi_success(
        MPI_Allreduce(
            &local_a, &a_min, 1,
            MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce for minimum snapshot scale factor");
    require_mpi_success(
        MPI_Allreduce(
            &local_a, &a_max, 1,
            MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for maximum snapshot scale factor");

    int index_min = 0;
    int index_max = 0;
    require_mpi_success(
        MPI_Allreduce(
            &snapshot_index, &index_min, 1,
            MPI_INT, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce for minimum snapshot index");
    require_mpi_success(
        MPI_Allreduce(
            &snapshot_index, &index_max, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for maximum snapshot index");
    if (a_min != a_max || index_min < 0 || index_min != index_max) {
        throw std::invalid_argument(
            "Parallel snapshot scale factor/index must agree across ranks");
    }

    const CollectiveMassRepresentation mass_representation =
        require_collective_mass_representation(local_particles);
    const bool write_explicit_masses = !mass_representation.uniform;

    std::string physics_fingerprint;
    std::string metadata_json;
    std::uint64_t metadata_dynamic_payload_bytes = 0U;
    std::optional<RunMetadata> prepared_run_metadata;
    int local_metadata_failed = 0;
    try {
        prepared_run_metadata.emplace(RunMetadata::from_config(
            config_, ProductLineage::DirectSimulation, {}, true));
        if (mpi_size > 1) {
            // One distributed snapshot is a shared artifact. Process-local FFT
            // plan identities therefore cannot be serialized as if every rank
            // observed the same execution provenance. Keep only the currently
            // defined collective FFTW-MPI policy record, and validate its
            // shared-artifact contract before allowing it to survive.
            auto& planning_records =
                prepared_run_metadata->fftw_planning_records;
            constexpr std::string_view shared_mpi_plan_scope =
                "planner_policy_only_no_collective_mpi_wisdom";
            for (const auto& record : planning_records) {
                if (record.backend != "fftw_mpi") continue;
                if (record.plan_identity_scope != shared_mpi_plan_scope
                    || !record.plan_representation_sha256.empty()
                    || record.mpi_rank_count
                        != static_cast<std::size_t>(mpi_size)) {
                    throw std::logic_error(
                        "Parallel snapshot encountered an FFTW-MPI planning record that is not valid shared-artifact provenance");
                }
            }
            const std::size_t omitted_record_count = std::erase_if(
                planning_records,
                [](const mesh::FftwPlanningRecord& record) {
                    // Fail closed for future or unknown backends. New planning
                    // backends must explicitly establish shared-artifact
                    // semantics before their records can enter global metadata.
                    return record.backend != "fftw_mpi";
                });
            if (omitted_record_count != 0U) {
                prepared_run_metadata->fftw_planning_records_complete = false;
            }
        }
        if (prepared_run_metadata->ic_mesh_per_dimension
            > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "Parallel snapshot inherited IC mesh exceeds uint32 storage");
        }
        metadata_dynamic_payload_bytes =
            run_metadata_dynamic_payload_bytes(*prepared_run_metadata);
        metadata_json = prepared_run_metadata->to_json();
        physics_fingerprint = snapshot_physics_fingerprint(config_);
        (void)snapshot_metadata_memory_plan(
            metadata_dynamic_payload_bytes,
            metadata_json.size(),
            physics_fingerprint.size());
    } catch (...) {
        local_metadata_failed = 1;
    }
    int any_metadata_failed = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_metadata_failed, &any_metadata_failed,
            1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for parallel snapshot metadata status");
    if (any_metadata_failed != 0) {
        throw std::runtime_error(
            "Parallel snapshot metadata construction failed on at least one rank");
    }
    if (!prepared_run_metadata.has_value()) {
        throw std::logic_error(
            "Parallel snapshot metadata collective succeeded without materialized metadata");
    }
    const RunMetadata& run_metadata = *prepared_run_metadata;
    require_exact_rank_string_agreement(
        physics_fingerprint, "Parallel snapshot physics fingerprint");
    require_exact_rank_string_agreement(
        metadata_json, "Parallel snapshot RunMetadata JSON");

    const std::uint64_t configured_batch = std::max<std::uint64_t>(
        1, config_.get_output().snapshot_batch_particles);
    std::uint64_t batch_min = 0;
    std::uint64_t batch_max = 0;
    require_mpi_success(
        MPI_Allreduce(
            &configured_batch, &batch_min, 1,
            MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce for minimum snapshot batch size");
    require_mpi_success(
        MPI_Allreduce(
            &configured_batch, &batch_max, 1,
            MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for maximum snapshot batch size");
    if (batch_min != batch_max) {
        throw std::invalid_argument(
            "Parallel snapshot batch size differs across ranks");
    }
    // MPI message counts are int. Coordinates and velocities carry three
    // scalars per particle, so this cap makes every payload count representable
    // without weakening the configured upper bound.
    constexpr std::uint64_t mpi_batch_limit =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 3);
    const std::uint64_t transfer_batch = std::min(
        configured_batch, mpi_batch_limit);

    const std::uint64_t local_count =
        static_cast<std::uint64_t>(local_owned);
    const std::uint64_t global_count =
        math::mpi_exact_uint64_sum(local_count);
    if (global_count != config_.num_particles()) {
        throw std::runtime_error(
            "Distributed snapshot global owned-particle count disagrees with configured N^3");
    }
    if (global_count
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
            "Parallel snapshot header exceeds uint32 particle count");
    }

    // The exact global sum above is at most uint32 here. Every
    // nonnegative partial prefix is therefore <= global_count, so the following
    // uint64 Exscan cannot wrap even though MPI supplies only a modular SUM.
    std::uint64_t offset = 0;
    require_mpi_success(
        MPI_Exscan(
            &local_count, &offset, 1,
            MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Exscan for parallel snapshot rank offset");
    if (rank == 0) offset = 0;
    const int local_offset_invalid =
        offset > global_count || local_count > global_count - offset ? 1 : 0;
    int any_offset_invalid = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_offset_invalid, &any_offset_invalid, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for parallel snapshot offsets");
    if (any_offset_invalid != 0) {
        throw std::runtime_error(
            "Parallel snapshot rank hyperslab exceeds global particle range");
    }

    std::vector<std::uint64_t> rank_counts;
    std::vector<std::uint64_t> rank_offsets;
    int local_topology_allocation_failed = 0;
    try {
        rank_counts.resize(static_cast<std::size_t>(mpi_size));
        rank_offsets.resize(static_cast<std::size_t>(mpi_size));
    } catch (...) {
        local_topology_allocation_failed = 1;
    }
    int any_topology_allocation_failed = 0;
    require_mpi_success(
        MPI_Allreduce(
            &local_topology_allocation_failed,
            &any_topology_allocation_failed,
            1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce for distributed snapshot topology allocation");
    if (any_topology_allocation_failed != 0) {
        throw std::runtime_error(
            "Distributed snapshot rank-topology allocation failed");
    }
    require_mpi_success(
        MPI_Allgather(
            &local_count, 1, MPI_UINT64_T,
            rank_counts.data(), 1, MPI_UINT64_T,
            MPI_COMM_WORLD),
        "MPI_Allgather for distributed snapshot rank counts");
    require_mpi_success(
        MPI_Allgather(
            &offset, 1, MPI_UINT64_T,
            rank_offsets.data(), 1, MPI_UINT64_T,
            MPI_COMM_WORLD),
        "MPI_Allgather for distributed snapshot rank offsets");

    std::uint64_t expected_rank_offset = 0;
    for (int source = 0; source < mpi_size; ++source) {
        const std::size_t source_index = static_cast<std::size_t>(source);
        if (rank_offsets[source_index] != expected_rank_offset
            || rank_counts[source_index]
                > global_count - expected_rank_offset) {
            throw std::runtime_error(
                "Distributed snapshot rank intervals are not an exact contiguous partition");
        }
        expected_rank_offset += rank_counts[source_index];
    }
    if (expected_rank_offset != global_count) {
        throw std::runtime_error(
            "Distributed snapshot rank intervals do not cover the global particle range");
    }

    const auto owned_ids = local_particles.get_ids().first(local_owned);
    domain::require_globally_unique_sorted_particle_ids(owned_ids);

    std::exception_ptr local_preflight_exception;
    BatchBuffers batch_buffers;
    try {
        validate_local_payload(
            local_particles, current_a, config_.get_box().L);
        const std::uint64_t capacity_count = rank == 0
            ? std::min(global_count, transfer_batch)
            : std::min(local_count, transfer_batch);
        const std::size_t capacity = static_cast<std::size_t>(capacity_count);
        batch_buffers.coordinates.resize(capacity * 3);
        batch_buffers.velocities.resize(capacity * 3);
        batch_buffers.ids.resize(capacity);
        if (write_explicit_masses) {
            batch_buffers.masses.resize(capacity);
        }
    } catch (...) {
        // Capturing the already-active exception is noexcept. In particular,
        // do not materialize a diagnostic string after a batch-buffer
        // allocation failure: a second allocation failure here could let one
        // rank escape before its peers enter the status collective.
        local_preflight_exception = std::current_exception();
    }
    synchronize_distributed_snapshot_exception(
        local_preflight_exception, mpi_size,
        "Parallel snapshot payload preflight");