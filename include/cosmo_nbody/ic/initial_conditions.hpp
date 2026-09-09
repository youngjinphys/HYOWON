// IC orchestration supports deterministic generated ICs and fail-closed native
// snapshot ingestion. Generated ICs apply the selected spectrum, Fourier support,
// 1LPT/2LPT displacement, lattice sampling, periodic wrapping, and stable IDs
// before transactional publication. Optional RealisedICEvidence records generated
// mode/phase-space/map diagnostics without asserting scientific accuracy or
// convergence. Snapshot ingestion accepts only explicit native units, velocity
// semantics, IDs, scale factor, and immutable content identity.
#pragma once

#include "cosmo_nbody/core/particle_store.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/cosmology/cosmology_model.hpp"
#include "cosmo_nbody/ic/realised_ic_evidence.hpp"

#include <optional>
#include <type_traits>

namespace cosmo_nbody {
namespace ic {

// Generated and snapshot IC paths construct complete local candidates and then
// publish them by move assignment. These operations must remain non-throwing;
// otherwise sequential publication of particle and evidence outputs would not
// provide the documented strong exception guarantee.
static_assert(std::is_nothrow_move_assignable_v<core::ParticleStore>);
static_assert(std::is_nothrow_move_assignable_v<
    std::optional<RealisedICEvidence>>);

class InitialConditions {
public:
    // Snapshot ingestion has no Fourier dependency. Keeping this path separate
    // prevents construction of an unused host FFT workspace.
    static void load_snapshot(
        const config::SimulationParameters& config,
        core::ParticleStore& particles);

    // Restart resume re-establishes immutable source-IC lineage without
    // allocating or rereading particle phase space. The configured source object
    // must still match its exact digest and descriptor physics.
    static void admit_snapshot_provenance(
        const config::SimulationParameters& config);

    // Generated ICs own exactly one FFT backend on the active IC mesh. The
    // evolution PM mesh is a distinct numerical coordinate and must not be
    // planned merely because an IC artifact is being generated. This keeps
    // memory use and FFT planning provenance bound to operations that actually
    // contribute to the generated particle state.
    static void generate(
        const config::SimulationParameters& config,
        const cosmology::CosmologyModel& cosmo,
        core::ParticleStore& particles,
        std::optional<RealisedICEvidence>* evidence_out = nullptr);
};

} // namespace ic
} // namespace cosmo_nbody
