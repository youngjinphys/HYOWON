#pragma once

#include <string>
#include <string_view>

namespace cosmo_nbody::runtime {

// Broadcast one arbitrary-length string while preserving collective call order
// when a receiver cannot allocate its payload buffer. All ranks either obtain the
// same value or leave the operation through an exception after a shared failure
// reduction; no rank enters the payload broadcast alone.
std::string broadcast_string_collective(
    std::string_view value,
    int source_rank,
    int rank,
    int size,
    const char* context);

} // namespace cosmo_nbody::runtime
