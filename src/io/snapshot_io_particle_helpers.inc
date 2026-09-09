
template <typename IdSpan>
void require_unique_ids_bounded(
    IdSpan ids,
    const char* context,
    bool caller_supplied_payload) {
    const auto throw_duplicate = [&]() -> void {
        const std::string message =
            std::string(context) + " contains duplicate stable ParticleIDs";
        if (caller_supplied_payload) {
            throw std::invalid_argument(message);
        }
        throw std::runtime_error(message);
    };
    const std::size_t n = ids.size();
    bool dense_range = true;
    for (const auto id : ids) {
        if (id >= static_cast<std::uint64_t>(n)) {
            dense_range = false;
            break;
        }
    }

    if (dense_range) {
        const std::size_t words = n / 64 + (n % 64 != 0 ? 1 : 0);
        std::vector<std::uint64_t> seen(words, 0);
        for (const auto id : ids) {
            const std::size_t index = static_cast<std::size_t>(id);
            const std::size_t word = index / 64;
            const std::uint64_t mask = 1ULL << (index % 64);
            if ((seen[word] & mask) != 0) {
                throw_duplicate();
            }
            seen[word] |= mask;
        }
        return;
    }

    // Irregular external IDs remain supported without unordered_set node
    // overhead. The fallback costs 8N contiguous bytes and is released
    // immediately after validation.
    std::vector<std::uint64_t> sorted(ids.begin(), ids.end());
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw_duplicate();
    }
}

void require_finite_particle_payload(
    const core::ParticleStore& particles,
    core::Real current_a,
    core::Real box_size) {
    if (!std::isfinite(current_a) || current_a <= 0.0) {
        throw std::invalid_argument(
            "Snapshot scale factor must be finite and positive");
    }
    const core::Real inv_a = 1.0 / current_a;
    if (!std::isfinite(inv_a) || inv_a <= 0.0) {
        throw std::overflow_error(
            "Snapshot inverse scale factor is not finite and positive");
    }

    const std::size_t n = particles.num_owned_particles();
    if (const auto uniform_mass = particles.get_uniform_mass();
        uniform_mass.has_value()
        && (!std::isfinite(*uniform_mass) || *uniform_mass <= 0.0)) {
        throw std::invalid_argument(
            "Snapshot uniform particle mass must be finite and positive");
    }

    for (std::size_t i = 0; i < n; ++i) {
        const core::Real x = particles.get_positions_x()[i];
        const core::Real y = particles.get_positions_y()[i];
        const core::Real z = particles.get_positions_z()[i];
        const core::Real px = particles.get_momenta_x()[i];
        const core::Real py = particles.get_momenta_y()[i];
        const core::Real pz = particles.get_momenta_z()[i];
        const core::Real mass = particles.mass_at(i);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
            || !std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) {
            throw std::invalid_argument(
                "Snapshot particle phase-space fields must be finite");
        }
        if (x < 0.0 || x >= box_size
            || y < 0.0 || y >= box_size
            || z < 0.0 || z >= box_size) {
            throw std::invalid_argument(
                "Snapshot coordinates must lie in the periodic half-open interval [0,L)");
        }
        if (!std::isfinite(px * inv_a)
            || !std::isfinite(py * inv_a)
            || !std::isfinite(pz * inv_a)) {
            throw std::overflow_error(
                "Snapshot momentum-to-velocity conversion overflowed");
        }
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument(
                "Snapshot particle masses must be finite and positive");
        }
    }
    require_unique_ids_bounded(
        particles.get_ids().first(n),
        "Snapshot payload",
        true);
}

void select_vector_hyperslab(
    hid_t file_space,
    std::size_t offset,
    std::size_t count) {
    const hsize_t start[2] = {
        static_cast<hsize_t>(offset), 0};
    const hsize_t block[2] = {
        static_cast<hsize_t>(count), 3};
    check_hdf5(
        H5Sselect_hyperslab(
            file_space,
            H5S_SELECT_SET,
            start,
            nullptr,
            block,
            nullptr),
        "select snapshot vector hyperslab");
}

void write_vector_dataset_batched(
    hid_t group,
    const std::string& name,
    hid_t type_real,
    std::size_t n,
    std::size_t batch_particles,
    const core::ParticleStore& particles,
    core::Real current_a,
    bool velocities) {
    const hsize_t dims[2] = {static_cast<hsize_t>(n), 3};
    auto file_space = H5SpaceHandle::checked(
        H5Screate_simple(2, dims, nullptr),
        "snapshot vector file space " + name);
    auto dataset = create_dataset(
        group, name, type_real, file_space.get());
    write_string_attr(
        dataset.get(),
        schema::ATTR_UNIT,
        velocities
            ? schema::VAL_VELOCITY_DATASET_UNIT
            : schema::VAL_COORDINATE_DATASET_UNIT);
    if (velocities) {
        write_string_attr(
            dataset.get(),
            schema::ATTR_VELOCITY_CONVENTION,
            schema::VAL_VELOCITY_CONVENTION);
    }

    std::vector<core::Real> buffer;
    if (batch_particles > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::overflow_error(
            "Snapshot batch vector size overflows size_t");
    }
    buffer.resize(batch_particles * 3);
    const core::Real inv_a = 1.0 / current_a;

    for (std::size_t offset = 0; offset < n; offset += batch_particles) {
        const std::size_t count = std::min(batch_particles, n - offset);
        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t index = offset + local;
            if (velocities) {
                buffer[3 * local + 0] =
                    particles.get_momenta_x()[index] * inv_a;
                buffer[3 * local + 1] =
                    particles.get_momenta_y()[index] * inv_a;
                buffer[3 * local + 2] =
                    particles.get_momenta_z()[index] * inv_a;
            } else {
                buffer[3 * local + 0] = particles.get_positions_x()[index];
                buffer[3 * local + 1] = particles.get_positions_y()[index];
                buffer[3 * local + 2] = particles.get_positions_z()[index];
            }
        }

        auto selected_file_space = H5SpaceHandle::checked(
            H5Dget_space(dataset.get()),
            "snapshot vector selected file space " + name);
        select_vector_hyperslab(
            selected_file_space.get(), offset, count);
        const hsize_t memory_dims[2] = {
            static_cast<hsize_t>(count), 3};
        auto memory_space = H5SpaceHandle::checked(
            H5Screate_simple(2, memory_dims, nullptr),
            "snapshot vector memory space " + name);
        check_hdf5(
            H5Dwrite(
                dataset.get(),
                type_real,
                memory_space.get(),
                selected_file_space.get(),
                H5P_DEFAULT,
                buffer.data()),
            "write snapshot vector batch " + name);
    }
}

void read_vector_dataset_batched(
    hid_t group,
    const std::string& name,
    hid_t type_real,
    std::size_t n,
    std::size_t batch_particles,
    core::ParticleStore& loaded,
    core::Real current_a,
    bool velocities) {
    auto dataset = open_dataset(group, name);
    require_vector_dataset_shape(dataset.get(), name, n);
    require_string_attr_value(
        dataset.get(),
        schema::ATTR_UNIT,
        velocities
            ? schema::VAL_VELOCITY_DATASET_UNIT
            : schema::VAL_COORDINATE_DATASET_UNIT,
        "Snapshot dataset " + name);
    if (velocities) {
        const std::string convention = read_string_attr(
            dataset.get(),
            schema::ATTR_VELOCITY_CONVENTION,
            schema::VAL_VELOCITY_CONVENTION.size());
        if (convention != schema::VAL_VELOCITY_CONVENTION) {
            throw std::runtime_error(
                "Unsupported velocity convention in snapshot: " + convention);
        }
    }

    std::vector<core::Real> buffer(batch_particles * 3);
    for (std::size_t offset = 0; offset < n; offset += batch_particles) {
        const std::size_t count = std::min(batch_particles, n - offset);
        auto file_space = H5SpaceHandle::checked(
            H5Dget_space(dataset.get()),
            "snapshot read vector file space " + name);
        select_vector_hyperslab(file_space.get(), offset, count);
        const hsize_t memory_dims[2] = {
            static_cast<hsize_t>(count), 3};
        auto memory_space = H5SpaceHandle::checked(
            H5Screate_simple(2, memory_dims, nullptr),
            "snapshot read vector memory space " + name);
        check_hdf5(
            H5Dread(
                dataset.get(),
                type_real,
                memory_space.get(),
                file_space.get(),
                H5P_DEFAULT,
                buffer.data()),
            "read snapshot vector batch " + name);

        for (std::size_t local = 0; local < count; ++local) {
            const std::size_t index = offset + local;
            const core::Real a0 = buffer[3 * local + 0];
            const core::Real a1 = buffer[3 * local + 1];
            const core::Real a2 = buffer[3 * local + 2];
            if (!std::isfinite(a0) || !std::isfinite(a1) || !std::isfinite(a2)) {
                throw std::runtime_error(
                    "Snapshot vector batch contains non-finite values: " + name);
            }
            if (velocities) {
                const core::Real p0 = a0 * current_a;
                const core::Real p1 = a1 * current_a;
                const core::Real p2 = a2 * current_a;
                if (!std::isfinite(p0) || !std::isfinite(p1)
                    || !std::isfinite(p2)) {
                    throw std::runtime_error(
                        "Snapshot velocity-to-momentum conversion overflowed");
                }
                loaded.get_momenta_x()[index] = p0;
                loaded.get_momenta_y()[index] = p1;
                loaded.get_momenta_z()[index] = p2;
            } else {
                loaded.get_positions_x()[index] = a0;
                loaded.get_positions_y()[index] = a1;
                loaded.get_positions_z()[index] = a2;
            }
        }
    }
}
