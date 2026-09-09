#pragma once

#include <string>
#include <string_view>

namespace cosmo_nbody::io::native_identity {

inline constexpr std::string_view SOFTWARE_NAME = "HYOWON";
inline constexpr std::string_view SOFTWARE_NAME_ATTRIBUTE = "SoftwareName";

// HDF5 helpers require std::string references; both objects derive from the
// canonical literals above so snapshot/restart admission cannot diverge.
inline const std::string SOFTWARE_NAME_STRING{SOFTWARE_NAME};
inline const std::string SOFTWARE_NAME_ATTRIBUTE_STRING{SOFTWARE_NAME_ATTRIBUTE};

} // namespace cosmo_nbody::io::native_identity
