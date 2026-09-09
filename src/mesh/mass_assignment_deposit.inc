#include "cosmo_nbody/runtime/thread_policy.hpp"

CICMassAssignment::CICMassAssignment(
    const MeshGeometry& geom,
    config::MemoryPolicyParams memory_policy,
    core::Real deposition_mass_unit)
    : CICMassAssignment(geom, std::move(memory_policy)) {
    if (!std::isfinite(deposition_mass_unit) || deposition_mass_unit <= 0.0) {
        throw std::invalid_argument(
            "CIC deposition mass unit must be finite and positive");
    }
    deposition_mass_unit_ = deposition_mass_unit;
}

void CICMassAssignment::deposit_impl(
    std::span<const core::Real> pos_x,
    std::span<const core::Real> pos_y,
    std::span<const core::Real> pos_z,
    std::span<const core::Real> masses,
    std::optional<core::Real> uniform_mass,
    CICDepositWorkspace* workspace,
    RealField& field,
    RealField* right_ghost) const {
    require_position_sizes(pos_x, pos_y, pos_z);
    require_finite_positions(pos_x, pos_y, pos_z);

    if (field.size() != geom_.padded_real_size()) {
        throw std::invalid_argument(
            "CIC deposit field size does not match mesh geometry");
    }
    const std::size_t N = geom_.grid_size();
    if (N > std::numeric_limits<std::size_t>::max() / N) {
        throw std::overflow_error("CIC ghost-plane size overflows size_t");
    }
    const std::size_t ghost_plane_size = N * N;
    auto validate_ghost = [ghost_plane_size](
        const RealField* ghost,
        const char* label) {
        if (ghost && ghost->size() != ghost_plane_size) {
            throw std::invalid_argument(
                std::string("CIC ") + label
                + " ghost plane has incorrect size");
        }
    };
    validate_ghost(right_ghost, "right");

    const auto scale_mass = [this](core::Real mass) {
        return deposition_mass_unit_ == core::Real{1.0}
            ? mass
            : mass / deposition_mass_unit_;
    };

    if (uniform_mass.has_value()) {
        if (!std::isfinite(*uniform_mass) || *uniform_mass < 0.0) {
            throw std::invalid_argument(
                "CIC uniform mass must be finite and non-negative");
        }
        const core::Real deposited_mass = scale_mass(*uniform_mass);
        if (!std::isfinite(deposited_mass)
            || (*uniform_mass > 0.0 && deposited_mass <= 0.0)) {
            throw std::overflow_error(
                "CIC uniform mass is not representable in the requested deposition unit");
        }
    } else {
        if (masses.size() != pos_x.size()) {
            throw std::invalid_argument(
                "CIC non-uniform mass array size mismatch");
        }
        int invalid_mass = 0;
        int invalid_scaled_mass = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static) \
            reduction(|: invalid_mass, invalid_scaled_mass) \
            if(runtime::should_use_host_parallel_team(masses.size()))
#endif
        for (std::size_t index = 0; index < masses.size(); ++index) {
            const core::Real mass = masses[index];
            if (!std::isfinite(mass) || mass < 0.0) {
                invalid_mass = 1;
                continue;
            }
            const core::Real deposited_mass = scale_mass(mass);
            if (!std::isfinite(deposited_mass)
                || (mass > 0.0 && deposited_mass <= 0.0)) {
                invalid_scaled_mass = 1;
            }
        }
        if (invalid_mass != 0) {
            throw std::invalid_argument(
                "CIC masses must be finite and non-negative");
        }
        if (invalid_scaled_mass != 0) {
            throw std::overflow_error(
                "CIC mass is not representable in the requested deposition unit");
        }
    }

    const core::Real L = geom_.box_size();
    const std::size_t n_part = pos_x.size();
    const std::size_t local_planes = geom_.local_n0();
    const std::size_t local_start = geom_.local_0_start();
    const std::size_t local_end = local_start + local_planes;
    const bool decomposed = local_planes < N;

    // Particle ownership is defined by the wrapped base x-cell. A CIC particle
    // contributes to that cell and the cell immediately to its right, so a
    // decomposed slab needs exactly one outgoing right-overflow plane. Requiring
    // that plane avoids silently dropping upper-cell mass.
    if (decomposed && !right_ghost) {
        throw std::invalid_argument(
            "Decomposed CIC deposition requires a right overflow plane");
    }

    if (n_part == 0) {
        zero_buffer(std::span<core::Real>(field.data(), field.size()));
        if (right_ghost) {
            zero_buffer(std::span<core::Real>(
                right_ghost->data(), right_ghost->size()));
        }
        return;
    }

    const std::size_t right_task = decomposed && right_ghost ? 1 : 0;
    const std::size_t plane_tasks = local_planes + right_task;
    if (plane_tasks
        > static_cast<std::size_t>(
            std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error(
            "CIC plane task count exceeds uint64_t range");
    }
    std::uint64_t maximum_threads = 1;
#ifdef COSMO_NBODY_HAS_OPENMP
    maximum_threads = static_cast<std::uint64_t>(
        std::max(1, omp_get_max_threads()));
#endif
    const CICDepositMemoryPlan memory_plan = cic_deposit_memory_plan(
        static_cast<std::uint64_t>(n_part),
        static_cast<std::uint64_t>(N),
        static_cast<std::uint64_t>(plane_tasks),
        maximum_threads);

    XPlaneStorage call_local_storage;
    XPlaneStorage& bin_storage = workspace
        ? workspace->impl_->bins
        : call_local_storage;
    const XPlaneBins bins = build_x_plane_bins(
        pos_x, L, dx_, geom_, memory_policy_, memory_plan, bin_storage);

    // Counting-sort establishes destination-plane ownership for deposition. The
    // final offset must account for every input particle exactly once before any
    // output buffer is mutated. This is stronger than inferring coverage later
    // from a floating-point total-mass tolerance.
    if (bins.offsets.size() != N + 1
        || bins.offsets.front() != 0
        || bins.offsets.back() != n_part) {
        throw std::logic_error(
            "CIC x-plane binning does not cover the complete particle population");
    }

    // Fail before mutating any output when a caller supplies particles owned by
    // another slab. Without this admission check those particles would either
    // disappear or be partially accumulated into an unrelated periodic plane.
    if (decomposed) {
        for (std::size_t base_x = 0; base_x < N; ++base_x) {
            if (base_x >= local_start && base_x < local_end) continue;
            if (bins.offsets[base_x] != bins.offsets[base_x + 1]) {
                throw std::invalid_argument(
                    "CIC particle base cell is outside the local mesh slab");
            }
        }
    }

    zero_buffer(std::span<core::Real>(field.data(), field.size()));
    if (right_ghost) {
        zero_buffer(std::span<core::Real>(
            right_ghost->data(), right_ghost->size()));
    }

    const auto resolve_task = [&](std::size_t task) {
        if (task < local_planes) {
            const std::size_t local_x = task;
            return std::pair<std::size_t, core::Real*>{
                local_start + local_x,
                field.data() + local_x * ghost_plane_size};
        }
        return std::pair<std::size_t, core::Real*>{
            geom_.wrap_index(
                static_cast<std::int64_t>(local_end)),
            right_ghost->data()};
    };

    const auto deposit_plane_task = [&](std::size_t task) {
        const auto [global_x_bound, plane_bound] = resolve_task(task);
        const std::size_t global_x = global_x_bound;
        core::Real* const plane = plane_bound;
        const auto accumulate_source_bin =
            [&](std::size_t base_x, bool upper_x_weight) {
                for (std::size_t cursor = bins.offsets[base_x];
                     cursor < bins.offsets[base_x + 1];
                     ++cursor) {
                    const std::size_t particle = bins.particles.at(cursor);
                    const auto x_coordinate = cic_coordinate_1d(
                        pos_x[particle], L, dx_, N);
                    const auto y_coordinate = cic_coordinate_1d(
                        pos_y[particle], L, dx_, N);
                    const auto z_coordinate = cic_coordinate_1d(
                        pos_z[particle], L, dx_, N);
                    const auto iy = y_coordinate.base;
                    const auto iz = z_coordinate.base;
                    const core::Real tx = x_coordinate.fraction;
                    const core::Real ty = y_coordinate.fraction;
                    const core::Real tz = z_coordinate.fraction;
                    const core::Real x_weight =
                        upper_x_weight ? tx : (1.0 - tx);
                    const core::Real wy[2] = {1.0 - ty, ty};
                    const core::Real wz[2] = {1.0 - tz, tz};
                    const core::Real raw_mass = uniform_mass.has_value()
                        ? *uniform_mass : masses[particle];
                    const core::Real mass = scale_mass(raw_mass);

                    for (int y_offset = 0; y_offset < 2; ++y_offset) {
                        const std::size_t gy =
                            geom_.wrap_index(iy + y_offset);
                        for (int z_offset = 0; z_offset < 2; ++z_offset) {
                            const std::size_t gz =
                                geom_.wrap_index(iz + z_offset);
                            plane[gy * N + gz] +=
                                mass * x_weight
                                * wy[y_offset] * wz[z_offset];
                        }
                    }
                }
            };

        // A destination plane receives lower-x contributions from particles
        // based in that plane and upper-x contributions from the previous plane.
        accumulate_source_bin(global_x, false);
        accumulate_source_bin(
            geom_.wrap_index(
                static_cast<std::int64_t>(global_x) - 1),
            true);
    };

    std::vector<std::uint64_t> task_particle_visits(plane_tasks, 0);
    std::uint64_t total_particle_visits = 0;
    for (std::size_t task = 0; task < plane_tasks; ++task) {
        const auto [global_x, plane] = resolve_task(task);
        (void)plane;
        const std::size_t previous_x = geom_.wrap_index(
            static_cast<std::int64_t>(global_x) - 1);
        const std::size_t visits_size =
            bins.offsets[global_x + 1] - bins.offsets[global_x]
            + bins.offsets[previous_x + 1] - bins.offsets[previous_x];
        if (visits_size
            > static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max())) {
            throw std::overflow_error(
                "CIC plane particle visits exceed uint64_t range");
        }
        const std::uint64_t visits =
            static_cast<std::uint64_t>(visits_size);
        if (visits > std::numeric_limits<std::uint64_t>::max()
                         - total_particle_visits) {
            throw std::overflow_error(
                "CIC total particle visits overflow uint64_t");
        }
        task_particle_visits[task] = visits;
        total_particle_visits += visits;
    }

    // Each locally owned CIC particle has exactly two x-plane visits: its lower
    // base plane and its upper neighbour. Each visit then contributes the four
    // y-z products, giving the usual eight 3-D CIC weights. Verify the integer
    // routing identity directly instead of relying on a floating mass checksum
    // whose necessary roundoff allowance grows with particle count.
    if (n_part > static_cast<std::size_t>(
                     std::numeric_limits<std::uint64_t>::max() / 2)) {
        throw std::overflow_error(
            "CIC expected particle-visit count overflows uint64_t");
    }
    const std::uint64_t expected_particle_visits =
        2ULL * static_cast<std::uint64_t>(n_part);
    if (total_particle_visits != expected_particle_visits) {
        throw std::logic_error(
            "CIC destination-plane routing does not cover each particle exactly twice");
    }

    if (memory_plan.active_plane_worker_count
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "CIC active plane worker count exceeds size_t");
    }
    const std::size_t planned_workers = static_cast<std::size_t>(
        memory_plan.active_plane_worker_count);
    if (planned_workers == 0
        || planned_workers > plane_tasks
        || planned_workers > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::logic_error(
            "CIC memory plan returned an invalid active worker count");
    }

    std::vector<std::size_t> worker_boundaries;
#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel num_threads(static_cast<int>(planned_workers)) \
        if(planned_workers > 1)
    {
        const std::size_t worker = static_cast<std::size_t>(
            omp_get_thread_num());
        #pragma omp single
        {
            worker_boundaries = detail::weighted_contiguous_plane_boundaries(
                task_particle_visits,
                static_cast<std::size_t>(omp_get_num_threads()));
        }
        for (std::size_t task = worker_boundaries[worker];
             task < worker_boundaries[worker + 1];
             ++task) {
            deposit_plane_task(task);
        }
    }
#else
    worker_boundaries = detail::weighted_contiguous_plane_boundaries(
        task_particle_visits, 1);
    for (std::size_t task = 0; task < plane_tasks; ++task) {
        deposit_plane_task(task);
    }
#endif
}
