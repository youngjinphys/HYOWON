#pragma once

#include <hdf5.h>

namespace cosmo_nbody::io {

// Namespace-local overloads selected by unqualified HDF5 opens in restart
// readers. They reject links or payload storage outside the opened file.
hid_t H5Gopen2(hid_t location, const char* name, hid_t access_properties);
hid_t H5Dopen2(hid_t location, const char* name, hid_t access_properties);

} // namespace cosmo_nbody::io
