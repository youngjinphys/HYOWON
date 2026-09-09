#include "cosmo_nbody/ic/lpt_displacement.hpp"
#include "cosmo_nbody/ic/particle_lattice_bandlimit.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"
#include "cosmo_nbody/runtime/raw_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/real_scratch_buffer.hpp"
#include "cosmo_nbody/runtime/thread_policy.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody {
namespace ic {

namespace {

std::size_t checked_particle_count(std::size_t n) {
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error("LPT N^2 overflows size_t");
    }
    const std::size_t n2 = n * n;
    if (n != 0 && n2 > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error("LPT N^3 overflows size_t");
    }
    return n2 * n;
}

std::size_t checked_complex_bytes(std::size_t elements) {
    if (elements > std::numeric_limits<std::size_t>::max()
            / sizeof(std::complex<core::Real>)) {
        throw std::overflow_error(
            "LPT complex scratch byte size overflows size_t");
    }
    return elements * sizeof(std::complex<core::Real>);
}

#ifdef COSMO_NBODY_HAS_OPENMP
std::size_t square_work_items(std::size_t n) noexcept {
    if (n == 0) return 0;
    if (n > std::numeric_limits<std::size_t>::max() / n) {
        return std::numeric_limits<std::size_t>::max();
    }
    return n * n;
}
#endif

void uninitialized_copy_complex_parallel(
    const std::complex<core::Real>* source,
    std::size_t count,
    std::complex<core::Real>* destination) {
    using Complex = std::complex<core::Real>;
    static_assert(
        std::is_nothrow_copy_constructible_v<Complex>,
        "Parallel scratch preservation must not throw across OpenMP workers");

#ifdef COSMO_NBODY_HAS_OPENMP
    #pragma omp parallel if(runtime::should_use_host_parallel_team(count))
    {
        const std::size_t thread_count =
            static_cast<std::size_t>(omp_get_num_threads());
        const std::size_t thread_id =
            static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t base_count = count / thread_count;
        const std::size_t remainder = count % thread_count;
        const std::size_t local_count =
            base_count + (thread_id < remainder ? 1 : 0);
        const std::size_t begin =
            thread_id * base_count + std::min(thread_id, remainder);
        std::uninitialized_copy_n(
            source + begin,
            local_count,
            destination + begin);
    }
#else
    std::uninitialized_copy_n(source, count, destination);
#endif
}

bool is_nyquist(std::size_t index, std::size_t n) noexcept {
    return n % 2 == 0 && index == n / 2;
}

core::Real odd_derivative_component(
    core::Real full_k,
    bool nyquist) noexcept {
    return nyquist ? core::Real{0.0} : full_k;
}

core::Real hessian_numerator(
    int dim_a,
    int dim_b,
    const core::Real full_k[3],
    const bool nyquist[3]) noexcept {
    if (dim_a == dim_b) {
        return full_k[dim_a] * full_k[dim_a];
    }
    if (nyquist[dim_a] != nyquist[dim_b]) {
        return 0.0;
    }
    return full_k[dim_a] * full_k[dim_b];
}

bool apply_lpt_component_update(
    core::Real psi,
    core::Real position_factor,
    core::Real momentum_factor,
    core::Real box_size,
    core::Real& position,
    core::Real& momentum) noexcept {
    const core::Real displacement = position_factor * psi;
    const core::Real momentum_increment = momentum_factor * psi;
    if (!std::isfinite(displacement)
        || !std::isfinite(momentum_increment)) {
        return false;
    }

    const core::Real position_candidate = position + displacement;
    const core::Real momentum_candidate = momentum + momentum_increment;
    if (!std::isfinite(position_candidate)
        || !std::isfinite(momentum_candidate)) {
        return false;
    }

    const core::Real wrapped_position = math::wrap(
        position_candidate, box_size);
    if (!std::isfinite(wrapped_position)) {
        return false;
    }

    position = wrapped_position;
    momentum = momentum_candidate;
    return true;
}

void require_component_sizes(
    std::size_t expected,
    std::span<core::Real> pos_x,
    std::span<core::Real> pos_y,
    std::span<core::Real> pos_z,
    std::span<core::Real> mom_x,
    std::span<core::Real> mom_y,
    std::span<core::Real> mom_z) {
    if (pos_x.size() != expected
        || pos_y.size() != expected
        || pos_z.size() != expected
        || mom_x.size() != expected
        || mom_y.size() != expected
        || mom_z.size() != expected) {
        throw std::invalid_argument(
            "LPT component arrays must all contain exactly N^3 elements");
    }
}

void apply_to_arrays_impl(
    const config::SimulationParameters& config,
    const cosmology::CosmologyModel& cosmology,
    mesh::FFTBackend& fft,
    const mesh::ComplexField& density_k_field,
    mesh::ComplexField* reusable_density_storage,
    bool density_storage_file_backed,
    std::span<core::Real> pos_x,
    std::span<core::Real> pos_y,
    std::span<core::Real> pos_z,
    std::span<core::Real> mom_x,
    std::span<core::Real> mom_y,
    std::span<core::Real> mom_z) {
    const std::size_t N =
        static_cast<std::size_t>(config.get_box().N);
    // LPT uses the supplied Fourier mesh; the evolution PM mesh is an independent
    // coordinate and does not define IC generation resolution.
    const std::size_t N_mesh = fft.grid_size();
    const core::Real L = config.get_box().L;
    const std::size_t nz_complex = N_mesh / 2 + 1;
    const mesh::MeshGeometry geometry(L, N_mesh);

    if (N == 0 || N_mesh == 0) {
        throw std::invalid_argument(
            "LPT particle and mesh dimensions must be positive");
    }
    if (density_k_field.size() != geometry.complex_size()) {
        throw std::invalid_argument(
            "LPT density spectrum size does not match active FFT mesh geometry");
    }
    if (reusable_density_storage
        && reusable_density_storage->size() != geometry.complex_size()) {
        throw std::invalid_argument(
            "Reusable LPT density storage size does not match mesh geometry");
    }
    if (density_storage_file_backed
        && (!reusable_density_storage
            || !config::uses_file_backed_scratch(
                config.get_memory_policy().ic_scratch_mode))) {
        throw std::invalid_argument(
            "Already-file-backed LPT density reuse requires reusable storage and explicit disk scratch mode");
    }
    const std::size_t expected_particles = checked_particle_count(N);
    require_component_sizes(
        expected_particles,
        pos_x, pos_y, pos_z,
        mom_x, mom_y, mom_z);
    if (N_mesh % N != 0) {
        throw std::invalid_argument(
            "Current LPT lattice sampling requires the active FFT mesh to be an integer multiple of N");
    }

    const core::Real a_start =
        1.0 / (1.0 + config.get_time().z_start);
    const core::Real D1 = cosmology.D1(a_start);
    const core::Real D2 = cosmology.D2(a_start);
    const core::Real f1 = cosmology.f1(a_start);
    const core::Real f2 = cosmology.f2(a_start);
    const core::Real H_start = cosmology.H(a_start);

    if (!std::isfinite(a_start) || a_start <= 0.0
        || !std::isfinite(D1) || D1 <= 0.0
        || !std::isfinite(D2) || D2 >= 0.0
        || !std::isfinite(f1) || !std::isfinite(f2)
        || !std::isfinite(H_start) || H_start <= 0.0) {
        throw std::runtime_error(
            "Invalid growth or expansion factors for LPT");
    }

    const core::Real x_factor_1 = 1.0;
    const core::Real p_factor_1 =
        a_start * a_start * H_start * f1;
    const core::Real d1_squared = D1 * D1;
    if (!std::isfinite(d1_squared) || d1_squared <= 0.0) {
        throw std::runtime_error(
            "Invalid D1^2 for start-scaled 2LPT normalization");
    }
    const core::Real x_factor_2 = D2 / d1_squared;
    const core::Real p_factor_2 =
        a_start * a_start * H_start * f2 * x_factor_2;
    if (!std::isfinite(p_factor_1)
        || !std::isfinite(x_factor_2)
        || !std::isfinite(p_factor_2)) {
        throw std::runtime_error(
            "Invalid start-scaled LPT normalization");
    }

    // In Disk mode preserve the consumable density spectrum in file-backed
    // storage; backing placement does not change Fourier values, ordering, or LPT operators.
    std::unique_ptr<runtime::RawScratchBuffer> preserved_spectrum_storage;
    std::unique_ptr<mesh::ComplexField> preserved_spectrum_view;
    const mesh::ComplexField* first_order_source = &density_k_field;
    if (reusable_density_storage
        && !density_storage_file_backed
        && config::uses_file_backed_scratch(
            config.get_memory_policy().ic_scratch_mode)) {
        static_assert(
            std::is_trivially_destructible_v<std::complex<core::Real>>,
            "External complex scratch relies on trivial destruction");
        preserved_spectrum_storage =
            std::make_unique<runtime::RawScratchBuffer>(
                checked_complex_bytes(geometry.complex_size()),
                config::ScratchMode::Disk,
                config.get_memory_policy().scratch_directory,
                "lpt_preserved_spectrum");
        auto* preserved = static_cast<std::complex<core::Real>*>(
            preserved_spectrum_storage->data());
        uninitialized_copy_complex_parallel(
            density_k_field.data(), geometry.complex_size(), preserved);
        preserved_spectrum_view = std::make_unique<mesh::ComplexField>(
            preserved,
            geometry.complex_size(),
            mesh::external_mesh_storage);
        first_order_source = preserved_spectrum_view.get();

        // Release the original density allocation; first_order_source now refers
        // to the preserved spectrum.
        *reusable_density_storage = mesh::ComplexField(0);
    }

    // FFTW c2r destroys psi_k. Both work arrays follow the configured scratch
    // policy; their backing owners outlive the non-owning mesh views.
    const auto& scratch_policy = config.get_memory_policy();
    std::unique_ptr<runtime::RawScratchBuffer> psi_k_backing;
    std::unique_ptr<runtime::RealScratchBuffer> psi_real_backing;
    std::unique_ptr<mesh::ComplexField> psi_k_owner;
    std::unique_ptr<mesh::RealField> psi_real_owner;
    if (config::uses_file_backed_scratch(scratch_policy.ic_scratch_mode)) {
        static_assert(std::is_trivially_destructible_v<std::complex<core::Real>>);
        psi_k_backing = std::make_unique<runtime::RawScratchBuffer>(
            checked_complex_bytes(geometry.complex_size()),
            scratch_policy.ic_scratch_mode, scratch_policy.scratch_directory,
            "lpt_psi_k");
        auto* data = static_cast<std::complex<core::Real>*>(psi_k_backing->data());
        std::uninitialized_value_construct_n(data, geometry.complex_size());
        psi_k_owner = std::make_unique<mesh::ComplexField>(
            data, geometry.complex_size(), mesh::external_mesh_storage);
        psi_real_backing = std::make_unique<runtime::RealScratchBuffer>(
            geometry.real_size(), scratch_policy, "lpt_psi_real");
        psi_real_owner = std::make_unique<mesh::RealField>(
            psi_real_backing->data(), psi_real_backing->size(),
            mesh::external_mesh_storage);
    } else {
        psi_k_owner = std::make_unique<mesh::ComplexField>(geometry.complex_size());
        psi_real_owner = std::make_unique<mesh::RealField>(geometry.real_size());
    }
    auto& psi_k = *psi_k_owner;
    auto& psi_real = *psi_real_owner;
    std::clog << "[ic] lpt_work_scratch_mode="
              << config::scratch_mode_name(scratch_policy.ic_scratch_mode)
              << " real_elements=" << geometry.real_size()
              << " complex_elements=" << geometry.complex_size() << '\n';
    const std::size_t sampling_ratio = N_mesh / N;
#ifdef COSMO_NBODY_HAS_OPENMP
    const std::size_t mesh_plane_work = square_work_items(N_mesh);
#endif

    for (int dim = 0; dim < 3; ++dim) {
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static) \
            if(runtime::should_use_host_parallel_team(mesh_plane_work))
#endif
        for (std::size_t ix = 0; ix < N_mesh; ++ix) {
            for (std::size_t iy = 0; iy < N_mesh; ++iy) {
                const core::Real kx = geometry.k_component(ix);
                const core::Real ky = geometry.k_component(iy);
                for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                    const core::Real kz = geometry.k_component(iz);
                    const core::Real full_k[3] = {kx, ky, kz};
                    const bool nyquist[3] = {
                        is_nyquist(ix, N_mesh),
                        is_nyquist(iy, N_mesh),
                        is_nyquist(iz, N_mesh)};
                    const core::Real k2 = kx * kx + ky * ky + kz * kz;
                    const std::size_t index =
                        geometry.complex_index(ix, iy, iz);
                    if (k2 == 0.0) {
                        psi_k[index] = {0.0, 0.0};
                        continue;
                    }
                    const core::Real derivative_k =
                        odd_derivative_component(
                            full_k[dim], nyquist[dim]);
                    const core::Real factor = derivative_k / k2;
                    const std::complex<core::Real> delta =
                        (*first_order_source)[index];
                    psi_k[index] = {
                        -delta.imag() * factor,
                        delta.real() * factor};
                }
            }
        }

        fft.inverse(psi_k, psi_real);
        int invalid_displacement = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(3) \
            reduction(|:invalid_displacement) schedule(static) \
            if(runtime::should_use_host_parallel_team(expected_particles))
#endif
        for (std::size_t ix = 0; ix < N; ++ix) {
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const std::size_t particle_index =
                        (ix * N + iy) * N + iz;
                    const core::Real psi = psi_real[geometry.real_index(
                        ix * sampling_ratio,
                        iy * sampling_ratio,
                        iz * sampling_ratio)];
                    if (!std::isfinite(psi)) {
                        invalid_displacement = 1;
                        continue;
                    }
                    bool valid_update = false;
                    if (dim == 0) {
                        valid_update = apply_lpt_component_update(
                            psi, x_factor_1, p_factor_1, L,
                            pos_x[particle_index], mom_x[particle_index]);
                    } else if (dim == 1) {
                        valid_update = apply_lpt_component_update(
                            psi, x_factor_1, p_factor_1, L,
                            pos_y[particle_index], mom_y[particle_index]);
                    } else {
                        valid_update = apply_lpt_component_update(
                            psi, x_factor_1, p_factor_1, L,
                            pos_z[particle_index], mom_z[particle_index]);
                    }
                    invalid_displacement |= !valid_update;
                }
            }
        }
        if (invalid_displacement) {
            throw std::runtime_error(
                "1LPT particle update is non-finite or unrepresentable");
        }
    }

    if (config.get_ic().lpt_order != 2) return;

    auto compute_hessian = [&](int dim_a, int dim_b) {
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static) \
            if(runtime::should_use_host_parallel_team(mesh_plane_work))
#endif
        for (std::size_t ix = 0; ix < N_mesh; ++ix) {
            for (std::size_t iy = 0; iy < N_mesh; ++iy) {
                const core::Real kx = geometry.k_component(ix);
                const core::Real ky = geometry.k_component(iy);
                for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                    const core::Real kz = geometry.k_component(iz);
                    const core::Real full_k[3] = {kx, ky, kz};
                    const bool nyquist[3] = {
                        is_nyquist(ix, N_mesh),
                        is_nyquist(iy, N_mesh),
                        is_nyquist(iz, N_mesh)};
                    const core::Real k2 = kx * kx + ky * ky + kz * kz;
                    const std::size_t index =
                        geometry.complex_index(ix, iy, iz);
                    if (k2 == 0.0) {
                        psi_k[index] = {0.0, 0.0};
                        continue;
                    }
                    const core::Real numerator = hessian_numerator(
                        dim_a, dim_b, full_k, nyquist);
                    psi_k[index] = (*first_order_source)[index]
                        * (numerator / k2);
                }
            }
        }
        fft.inverse(psi_k, psi_real);
        int invalid_hessian = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(2) \
            reduction(|:invalid_hessian) schedule(static) \
            if(runtime::should_use_host_parallel_team(mesh_plane_work))
#endif
        for (std::size_t ix = 0; ix < N_mesh; ++ix) {
            for (std::size_t iy = 0; iy < N_mesh; ++iy) {
                for (std::size_t iz = 0; iz < N_mesh; ++iz) {
                    invalid_hessian |= !std::isfinite(
                        psi_real[geometry.real_index(ix, iy, iz)]);
                }
            }
        }
        if (invalid_hessian) {
            throw std::runtime_error(
                "2LPT first-order Hessian is non-finite");
        }
    };

    std::unique_ptr<mesh::ComplexField> owned_source;
    mesh::ComplexField* source_k = preserved_spectrum_view
        ? preserved_spectrum_view.get()
        : reusable_density_storage;
    if (!source_k) {
        owned_source = std::make_unique<mesh::ComplexField>(
            geometry.complex_size());
        source_k = owned_source.get();
    }

    {
        runtime::RealScratchBuffer source_r(
            geometry.real_size(),
            config.get_memory_policy(),
            "2lpt_source_r");

        // The diagonal contribution is the second elementary invariant of H:
        //   Hxx*Hyy + Hxx*Hzz + Hyy*Hzz
        //     = 0.5 * ((tr H)^2 - Hxx^2 - Hyy^2 - Hzz^2).
        // For the discrete operator above, tr H(k)=delta(k) for every non-zero
        // mode because sum_i k_i^2/k^2=1. Accumulating this invariant removes a
        // full diagonal-sum field. It adds one exact c2r transform of the input
        // density but changes neither the 2LPT equation nor its band limit.
        for (int dim = 0; dim < 3; ++dim) {
            compute_hessian(dim, dim);
#ifdef COSMO_NBODY_HAS_OPENMP
            #pragma omp parallel for collapse(2) schedule(static) \
                if(runtime::should_use_host_parallel_team(mesh_plane_work))
#endif
            for (std::size_t ix = 0; ix < N_mesh; ++ix) {
                for (std::size_t iy = 0; iy < N_mesh; ++iy) {
                    for (std::size_t iz = 0; iz < N_mesh; ++iz) {
                        const std::size_t compact =
                            geometry.real_index(ix, iy, iz);
                        const core::Real hessian =
                            psi_real[geometry.real_index(ix, iy, iz)];
                        const core::Real term =
                            core::Real{0.5} * hessian * hessian;
                        if (dim == 0) {
                            source_r[compact] = -term;
                        } else {
                            source_r[compact] -= term;
                        }
                    }
                }
            }
        }

        constexpr int cross_pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        for (const auto& pair : cross_pairs) {
            compute_hessian(pair[0], pair[1]);
#ifdef COSMO_NBODY_HAS_OPENMP
            #pragma omp parallel for collapse(2) schedule(static) \
                if(runtime::should_use_host_parallel_team(mesh_plane_work))
#endif
            for (std::size_t ix = 0; ix < N_mesh; ++ix) {
                for (std::size_t iy = 0; iy < N_mesh; ++iy) {
                    for (std::size_t iz = 0; iz < N_mesh; ++iz) {
                        const std::size_t compact =
                            geometry.real_index(ix, iy, iz);
                        const core::Real hessian =
                            psi_real[geometry.real_index(ix, iy, iz)];
                        source_r[compact] -= hessian * hessian;
                    }
                }
            }
        }

        // Complete the diagonal invariant with 0.5*(tr H)^2. The zero mode is
        // explicitly removed to match compute_hessian's k^2==0 convention.
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for schedule(static) \
            if(runtime::should_use_host_parallel_team(psi_k.size()))
#endif
        for (std::size_t index = 0; index < psi_k.size(); ++index) {
            psi_k[index] = (*first_order_source)[index];
        }
        psi_k[0] = {0.0, 0.0};
        fft.inverse(psi_k, psi_real);
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static) \
            if(runtime::should_use_host_parallel_team(mesh_plane_work))
#endif
        for (std::size_t ix = 0; ix < N_mesh; ++ix) {
            for (std::size_t iy = 0; iy < N_mesh; ++iy) {
                for (std::size_t iz = 0; iz < N_mesh; ++iz) {
                    const std::size_t compact =
                        geometry.real_index(ix, iy, iz);
                    const core::Real trace =
                        psi_real[geometry.real_index(ix, iy, iz)];
                    source_r[compact] +=
                        core::Real{0.5} * trace * trace;
                }
            }
        }

        int invalid_source = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for reduction(|:invalid_source) schedule(static) \
            if(runtime::should_use_host_parallel_team(source_r.size()))
#endif
        for (std::size_t index = 0; index < source_r.size(); ++index) {
            invalid_source |= !std::isfinite(source_r[index]);
        }
        if (invalid_source) {
            throw std::overflow_error(
                "2LPT quadratic source is non-finite");
        }

        // The original density spectrum is no longer needed; its reusable
        // storage may now hold the 2LPT source.
        fft.forward_external(
            source_r.data(),
            source_r.size(),
            *source_k);
    }

    apply_particle_lattice_bandlimit(
        geometry, N, *source_k, config.ic_effective_max_mode_per_axis());

    for (int dim = 0; dim < 3; ++dim) {
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static) \
            if(runtime::should_use_host_parallel_team(mesh_plane_work))
#endif
        for (std::size_t ix = 0; ix < N_mesh; ++ix) {
            for (std::size_t iy = 0; iy < N_mesh; ++iy) {
                const core::Real kx = geometry.k_component(ix);
                const core::Real ky = geometry.k_component(iy);
                for (std::size_t iz = 0; iz < nz_complex; ++iz) {
                    const core::Real kz = geometry.k_component(iz);
                    const core::Real full_k[3] = {kx, ky, kz};
                    const bool nyquist[3] = {
                        is_nyquist(ix, N_mesh),
                        is_nyquist(iy, N_mesh),
                        is_nyquist(iz, N_mesh)};
                    const core::Real k2 = kx * kx + ky * ky + kz * kz;
                    const std::size_t index =
                        geometry.complex_index(ix, iy, iz);
                    if (k2 == 0.0) {
                        psi_k[index] = {0.0, 0.0};
                        continue;
                    }
                    const core::Real derivative_k =
                        odd_derivative_component(
                            full_k[dim], nyquist[dim]);
                    const core::Real factor = derivative_k / k2;
                    const std::complex<core::Real> source = (*source_k)[index];
                    psi_k[index] = {
                        source.imag() * factor,
                        -source.real() * factor};
                }
            }
        }

        fft.inverse(psi_k, psi_real);
        int invalid_displacement = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
        #pragma omp parallel for collapse(3) \
            reduction(|:invalid_displacement) schedule(static) \
            if(runtime::should_use_host_parallel_team(expected_particles))
#endif
        for (std::size_t ix = 0; ix < N; ++ix) {
            for (std::size_t iy = 0; iy < N; ++iy) {
                for (std::size_t iz = 0; iz < N; ++iz) {
                    const std::size_t particle_index =
                        (ix * N + iy) * N + iz;
                    const core::Real psi = psi_real[geometry.real_index(
                        ix * sampling_ratio,
                        iy * sampling_ratio,
                        iz * sampling_ratio)];
                    if (!std::isfinite(psi)) {
                        invalid_displacement = 1;
                        continue;
                    }
                    bool valid_update = false;
                    if (dim == 0) {
                        valid_update = apply_lpt_component_update(
                            psi, x_factor_2, p_factor_2, L,
                            pos_x[particle_index], mom_x[particle_index]);
                    } else if (dim == 1) {
                        valid_update = apply_lpt_component_update(
                            psi, x_factor_2, p_factor_2, L,
                            pos_y[particle_index], mom_y[particle_index]);
                    } else {
                        valid_update = apply_lpt_component_update(
                            psi, x_factor_2, p_factor_2, L,
                            pos_z[particle_index], mom_z[particle_index]);
                    }
                    invalid_displacement |= !valid_update;
                }
            }
        }
        if (invalid_displacement) {
            throw std::runtime_error(
                "2LPT particle update is non-finite or unrepresentable");
        }
    }
}

} // namespace

void LPTDisplacement::apply_to_arrays(
    const config::SimulationParameters& config,
    const cosmology::CosmologyModel& cosmology,
    mesh::FFTBackend& fft,
    const mesh::ComplexField& density_k_field,
    std::span<core::Real> pos_x,
    std::span<core::Real> pos_y,
    std::span<core::Real> pos_z,
    std::span<core::Real> mom_x,
    std::span<core::Real> mom_y,
    std::span<core::Real> mom_z) {
    apply_to_arrays_impl(
        config,
        cosmology,
        fft,
        density_k_field,
        nullptr,
        false,
        pos_x, pos_y, pos_z,
        mom_x, mom_y, mom_z);
}

void LPTDisplacement::apply_to_arrays_reusing_density(
    const config::SimulationParameters& config,
    const cosmology::CosmologyModel& cosmology,
    mesh::FFTBackend& fft,
    mesh::ComplexField& density_k_field,
    std::span<core::Real> pos_x,
    std::span<core::Real> pos_y,
    std::span<core::Real> pos_z,
    std::span<core::Real> mom_x,
    std::span<core::Real> mom_y,
    std::span<core::Real> mom_z) {
    apply_to_arrays_impl(
        config,
        cosmology,
        fft,
        density_k_field,
        &density_k_field,
        false,
        pos_x, pos_y, pos_z,
        mom_x, mom_y, mom_z);
}

void LPTDisplacement::apply_to_arrays_reusing_file_backed_density(
    const config::SimulationParameters& config,
    const cosmology::CosmologyModel& cosmology,
    mesh::FFTBackend& fft,
    mesh::ComplexField& density_k_field,
    std::span<core::Real> pos_x,
    std::span<core::Real> pos_y,
    std::span<core::Real> pos_z,
    std::span<core::Real> mom_x,
    std::span<core::Real> mom_y,
    std::span<core::Real> mom_z) {
    apply_to_arrays_impl(
        config,
        cosmology,
        fft,
        density_k_field,
        &density_k_field,
        true,
        pos_x, pos_y, pos_z,
        mom_x, mom_y, mom_z);
}

} // namespace ic
} // namespace cosmo_nbody
