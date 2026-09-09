#pragma once

#include "cosmo_nbody/core/types.hpp"
#include "cosmo_nbody/config/simulation_parameters.hpp"
#include "cosmo_nbody/ic/linear_power_spectrum.hpp"
#include "cosmo_nbody/ic/philox_rng.hpp"
#include "cosmo_nbody/mesh/mesh_field.hpp"

#include <cstddef>
#include <complex>
#include <string_view>

namespace cosmo_nbody {
namespace ic {

class RandomField {
public:
    // Mapping identifier from seed, integer mode and stream to random draws.
    inline static constexpr std::string_view RNG_METHOD = rng::PHILOX_METHOD;

    // IC mesh is explicit and may differ from the evolution mesh; integer mode
    // addressing preserves shared representable modes across mesh choices.
    static void generate(
        const config::SimulationParameters& config,
        const LinearPowerSpectrum& pk,
        mesh::ComplexField& out_field,
        std::size_t mesh_per_dimension);

    // File-backed path for raw storage without live complex objects. The mode
    // loop constructs every element exactly once, avoiding zero-fill and copy.
    static void generate_into_uninitialized_storage(
        const config::SimulationParameters& config,
        const LinearPowerSpectrum& pk,
        std::complex<core::Real>* external_data,
        std::size_t external_size,
        std::size_t mesh_per_dimension);
};

} // namespace ic
} // namespace cosmo_nbody
