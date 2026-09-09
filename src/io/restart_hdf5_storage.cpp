#include "cosmo_nbody/io/restart_hdf5_storage.hpp"

#include "cosmo_nbody/io/hdf5_handle.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cosmo_nbody::io {
namespace {

void require_file_local_hard_link(
    hid_t location,
    const char* name,
    const char* object_kind) {
    if (name == nullptr || name[0] == '\0') {
        throw std::invalid_argument(
            std::string("Restart ") + object_kind + " name must not be empty");
    }
    H5L_info_t info{};
    if (::H5Lget_info(location, name, &info, H5P_DEFAULT) < 0) {
        throw std::runtime_error(
            std::string("Failed to inspect restart ") + object_kind
            + " link: " + name);
    }
    if (info.type != H5L_TYPE_HARD) {
        throw std::runtime_error(
            std::string("Restart ") + object_kind
            + " must be a file-local HDF5 hard link: " + name);
    }
}

bool link_exists(hid_t location, const char* name) {
    const htri_t exists = ::H5Lexists(location, name, H5P_DEFAULT);
    if (exists < 0) {
        throw std::runtime_error(
            std::string("Failed to inspect optional restart HDF5 link: ") + name);
    }
    return exists > 0;
}

void require_inline_dataset_storage(hid_t dataset, const char* name) {
    auto creation = H5PropertyHandle::checked(
        ::H5Dget_create_plist(dataset),
        std::string("restart dataset creation properties ") + name);
    const H5D_layout_t layout = ::H5Pget_layout(creation.get());
    if (layout == H5D_LAYOUT_ERROR) {
        throw std::runtime_error(
            std::string("Failed to inspect restart dataset layout: ") + name);
    }
    if (layout == H5D_VIRTUAL) {
        throw std::runtime_error(
            std::string("Restart dataset uses virtual storage outside the restart file: ")
            + name);
    }
    const int external_count = ::H5Pget_external_count(creation.get());
    if (external_count < 0) {
        throw std::runtime_error(
            std::string("Failed to inspect restart external dataset storage: ")
            + name);
    }
    if (external_count != 0) {
        throw std::runtime_error(
            std::string("Restart dataset uses external raw-data storage: ")
            + name);
    }
}

void validate_dataset(hid_t group, const char* name) {
    require_file_local_hard_link(group, name, "dataset");
    auto dataset = H5DatasetHandle::checked(
        ::H5Dopen2(group, name, H5P_DEFAULT),
        std::string("open restart dataset for storage validation ") + name);
    require_inline_dataset_storage(dataset.get(), name);
}

void validate_particle_payload_storage(hid_t file, hid_t particles) {
    static constexpr std::array<std::string_view, 7> required_datasets{
        "pos_x", "pos_y", "pos_z",
        "mom_x", "mom_y", "mom_z",
        "ParticleIDs",
    };
    for (const std::string_view name : required_datasets) {
        validate_dataset(particles, name.data());
    }
    if (link_exists(particles, "masses")) {
        validate_dataset(particles, "masses");
    }

    // Marker admission opens only /Particles; also bind the persisted
    // integrator and diagnostic groups to file-local objects.
    if (::H5Iget_type(file) == H5I_FILE) {
        require_file_local_hard_link(file, "/TimeStepper", "group");
        if (link_exists(file, "/Diagnostics")) {
            require_file_local_hard_link(file, "/Diagnostics", "group");
        }
    }
}

bool is_particles_group_name(const char* name) noexcept {
    if (name == nullptr) return false;
    return std::string_view(name) == "/Particles"
        || std::string_view(name) == "Particles";
}

} // namespace

hid_t H5Gopen2(
    hid_t location,
    const char* name,
    hid_t access_properties) {
    require_file_local_hard_link(location, name, "group");
    const hid_t group = ::H5Gopen2(location, name, access_properties);
    if (group < 0) return group;
    try {
        if (is_particles_group_name(name)) {
            validate_particle_payload_storage(location, group);
        }
    } catch (...) {
        (void)::H5Gclose(group);
        throw;
    }
    return group;
}

hid_t H5Dopen2(
    hid_t location,
    const char* name,
    hid_t access_properties) {
    require_file_local_hard_link(location, name, "dataset");
    const hid_t dataset = ::H5Dopen2(location, name, access_properties);
    if (dataset < 0) return dataset;
    try {
        require_inline_dataset_storage(dataset, name);
    } catch (...) {
        (void)::H5Dclose(dataset);
        throw;
    }
    return dataset;
}

} // namespace cosmo_nbody::io
