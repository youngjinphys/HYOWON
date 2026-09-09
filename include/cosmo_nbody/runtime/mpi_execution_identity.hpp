#pragma once

#include <string>

namespace cosmo_nbody::config {
class SimulationParameters;
}

namespace cosmo_nbody::runtime {

struct MpiExecutionIdentity {
    std::string configuration;
    int real_bytes{0};
    int communicator_size{0};
};

// Require identical numerical configuration and scalar precision on every rank;
// executable bytes are provenance, not part of this invariant.
void require_mpi_execution_identity_agreement(
    const MpiExecutionIdentity& local_identity);

// Establish MPI configuration/ABI agreement before distributed state is allocated.
class MpiExecutionIdentityAgreement {
public:
    MpiExecutionIdentityAgreement(
        const config::SimulationParameters& config,
        int rank,
        int size);
};

// Accept no value or one canonical lowercase SHA-256 across MPI_COMM_WORLD;
// this checks collective data consistency, not simulation accuracy.
std::string agree_optional_sha256_collective(
    const std::string& local_value,
    const char* context);

} // namespace cosmo_nbody::runtime
