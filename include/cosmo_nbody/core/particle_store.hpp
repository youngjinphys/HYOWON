// Particle storage (Structure of Arrays).
#pragma once

#include "cosmo_nbody/core/types.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cosmo_nbody {
namespace core {

enum class FieldValidity {
    VALID,
    STALE,
    INVALID
};

class ParticleStore {
public:
    ParticleStore() = default;

    // Positions, IDs, and explicit masses describe every local force source:
    // the owned prefix followed by the read-only ghost suffix. Momentum and
    // acceleration are integration state and therefore exist only for owned
    // particles. This keeps remote TreePM sources compact without introducing a
    // second source container or changing source ordering.
    std::span<Real> get_positions_x() { return positions_x; }
    std::span<Real> get_positions_y() { return positions_y; }
    std::span<Real> get_positions_z() { return positions_z; }

    std::span<Real> get_momenta_x() { return momenta_x; }
    std::span<Real> get_momenta_y() { return momenta_y; }
    std::span<Real> get_momenta_z() { return momenta_z; }

    std::span<Real> get_accelerations_x() {
        require_valid_acceleration_storage();
        return accelerations_x;
    }
    std::span<Real> get_accelerations_y() {
        require_valid_acceleration_storage();
        return accelerations_y;
    }
    std::span<Real> get_accelerations_z() {
        require_valid_acceleration_storage();
        return accelerations_z;
    }

    // Force builders request mutable acceleration storage before filling it.
    // Storage may be omitted while accelerations are invalid (for example during
    // generated-IC materialization); the first mutable access allocates all
    // three owned components transactionally after the IC scratch lifetime has
    // ended. Ghost sources never carry acceleration storage.
    std::span<Real> mutable_accelerations_x() {
        ensure_acceleration_storage();
        return accelerations_x;
    }
    std::span<Real> mutable_accelerations_y() {
        ensure_acceleration_storage();
        return accelerations_y;
    }
    std::span<Real> mutable_accelerations_z() {
        ensure_acceleration_storage();
        return accelerations_z;
    }

    std::span<Real> get_masses() {
        if (uniform_mass.has_value()) {
            throw std::logic_error(
                "Masses are uniform; per-particle mass storage is not available");
        }
        return masses;
    }

    std::span<ParticleId> get_ids() { return ids; }

    std::span<const Real> get_positions_x() const { return positions_x; }
    std::span<const Real> get_positions_y() const { return positions_y; }
    std::span<const Real> get_positions_z() const { return positions_z; }

    std::span<const Real> get_momenta_x() const { return momenta_x; }
    std::span<const Real> get_momenta_y() const { return momenta_y; }
    std::span<const Real> get_momenta_z() const { return momenta_z; }

    std::span<const Real> get_accelerations_x() const {
        require_valid_acceleration_storage();
        return accelerations_x;
    }
    std::span<const Real> get_accelerations_y() const {
        require_valid_acceleration_storage();
        return accelerations_y;
    }
    std::span<const Real> get_accelerations_z() const {
        require_valid_acceleration_storage();
        return accelerations_z;
    }

    std::span<const Real> get_masses() const {
        if (uniform_mass.has_value()) {
            throw std::logic_error(
                "Masses are uniform; per-particle mass storage is not available");
        }
        return masses;
    }
    std::span<const ParticleId> get_ids() const { return ids; }

    FieldValidity get_acceleration_validity() const { return accel_validity; }
    void set_acceleration_validity(FieldValidity value) {
        validate_storage_layout();
        if (value == FieldValidity::VALID && !acceleration_storage_enabled_) {
            throw std::logic_error(
                "Cannot mark accelerations valid while component storage is omitted");
        }
        accel_validity = value;
    }

    FieldValidity get_ghost_validity() const { return ghost_validity; }
    void set_ghost_validity(FieldValidity v) { ghost_validity = v; }

    const std::string& get_verified_snapshot_ic_sha256() const noexcept {
        return verified_snapshot_ic_sha256;
    }

    void set_verified_snapshot_ic_sha256(std::string value) {
        if (!value.empty()) {
            if (value.size() != 64) {
                throw std::invalid_argument(
                    "Verified snapshot IC SHA-256 must contain exactly 64 lowercase hexadecimal characters");
            }
            for (const char c : value) {
                const bool decimal = c >= '0' && c <= '9';
                const bool lowercase_hex = c >= 'a' && c <= 'f';
                if (!decimal && !lowercase_hex) {
                    throw std::invalid_argument(
                        "Verified snapshot IC SHA-256 must contain exactly 64 lowercase hexadecimal characters");
                }
            }
        }
        verified_snapshot_ic_sha256 = std::move(value);
    }

    std::optional<Real> get_uniform_mass() const { return uniform_mass; }
    void set_uniform_mass(std::optional<Real> value) {
        validate_storage_layout();
        if (value.has_value()
            && (!std::isfinite(*value) || *value <= 0.0)) {
            throw std::invalid_argument(
                "Uniform particle mass must be finite and positive");
        }

        if (value.has_value()) {
            uniform_mass = value;
            // clear() leaves capacity allocated. Release the obsolete per-source
            // mass storage so equal-mass mode actually removes the 8*Nsource
            // reservation, including any current ghost suffix.
            std::vector<Real>{}.swap(masses);
            return;
        }

        if (uniform_mass.has_value()) {
            // Expanding equal-mass storage must preserve every represented source
            // mass, including ghosts. Allocate before publishing the
            // representation change so failure leaves the uniform state intact.
            std::vector<Real> expanded(size(), *uniform_mass);
            masses.swap(expanded);
            uniform_mass.reset();
            return;
        }

        // Reasserting the already non-uniform representation is idempotent.
        masses.resize(size());
    }

    bool has_acceleration_storage() const {
        validate_storage_layout();
        return acceleration_storage_enabled_;
    }

    void ensure_acceleration_storage() {
        validate_storage_layout();
        if (acceleration_storage_enabled_) return;

        // Allocate all owned components before publishing any of them, so an
        // allocation failure cannot leave a partially materialized field.
        std::vector<Real> new_x(num_owned_particles_);
        std::vector<Real> new_y(num_owned_particles_);
        std::vector<Real> new_z(num_owned_particles_);
        accelerations_x.swap(new_x);
        accelerations_y.swap(new_y);
        accelerations_z.swap(new_z);
        acceleration_storage_enabled_ = true;
    }

    void release_accelerations() {
        validate_storage_layout();
        if (accel_validity == FieldValidity::VALID) {
            throw std::logic_error(
                "Cannot release acceleration storage while accelerations are valid");
        }
        std::vector<Real>{}.swap(accelerations_x);
        std::vector<Real>{}.swap(accelerations_y);
        std::vector<Real>{}.swap(accelerations_z);
        acceleration_storage_enabled_ = false;
        accel_validity = FieldValidity::INVALID;
    }

    // Total local force-source count (owned prefix + ghost suffix).
    std::size_t size() const { return positions_x.size(); }
    std::size_t num_owned_particles() const { return num_owned_particles_; }
    std::size_t num_ghost_particles() const { return size() - num_owned_particles_; }

    Real mass_at(std::size_t index) const {
        if (index >= size()) throw std::out_of_range("Particle mass index out of range");
        return uniform_mass.has_value() ? *uniform_mass : masses[index];
    }

    // Public resize materializes an owned-only store. Ghost suffixes must be
    // retired explicitly first so no source lacking momentum can accidentally be
    // promoted into integration state.
    void resize(std::size_t n) {
        validate_storage_layout();
        if (num_owned_particles_ != size()) {
            throw std::logic_error(
                "ParticleStore resize requires ghost sources to be cleared first");
        }
        if (n > size()) {
            positions_x.reserve(n);
            positions_y.reserve(n);
            positions_z.reserve(n);
            momenta_x.reserve(n);
            momenta_y.reserve(n);
            momenta_z.reserve(n);
            if (!uniform_mass.has_value()) masses.reserve(n);
            ids.reserve(n);
        }

        positions_x.resize(n);
        positions_y.resize(n);
        positions_z.resize(n);
        momenta_x.resize(n);
        momenta_y.resize(n);
        momenta_z.resize(n);
        if (!uniform_mass.has_value()) masses.resize(n);
        ids.resize(n);

        std::vector<Real>{}.swap(accelerations_x);
        std::vector<Real>{}.swap(accelerations_y);
        std::vector<Real>{}.swap(accelerations_z);
        acceleration_storage_enabled_ = false;

        num_owned_particles_ = n;
        accel_validity = FieldValidity::INVALID;
        ghost_validity = FieldValidity::INVALID;
    }

    // Domain migration rebuilds owned phase space while acceleration is already
    // invalid. Reuse the three acceleration allocations when their capacities
    // can represent the new owned population; stale values are deliberately not
    // preserved and must be overwritten by the next force refresh. If a rank
    // grows beyond the retained capacity, omit the buffers before resizing so a
    // replacement allocation cannot overlap the old three-component payload.
    void resize_owned_reusing_acceleration_storage(std::size_t n) {
        validate_storage_layout();
        if (num_owned_particles_ != size()) {
            throw std::logic_error(
                "ParticleStore owned-storage reuse requires ghosts to be cleared first");
        }
        if (accel_validity == FieldValidity::VALID) {
            throw std::logic_error(
                "ParticleStore cannot resize owned storage while accelerations are valid");
        }

        const bool reuse_accelerations = acceleration_storage_enabled_
            && accelerations_x.capacity() >= n
            && accelerations_y.capacity() >= n
            && accelerations_z.capacity() >= n;
        if (acceleration_storage_enabled_ && !reuse_accelerations) {
            std::vector<Real>{}.swap(accelerations_x);
            std::vector<Real>{}.swap(accelerations_y);
            std::vector<Real>{}.swap(accelerations_z);
            acceleration_storage_enabled_ = false;
        }

        // If ownership growth cannot reuse acceleration capacity, the old
        // three-component payload is already gone before any primary SoA reserve
        // may allocate a replacement. This prevents those two large generations
        // from overlapping at the migration publication peak.
        if (n > size()) {
            positions_x.reserve(n);
            positions_y.reserve(n);
            positions_z.reserve(n);
            momenta_x.reserve(n);
            momenta_y.reserve(n);
            momenta_z.reserve(n);
            if (!uniform_mass.has_value()) masses.reserve(n);
            ids.reserve(n);
        }

        positions_x.resize(n);
        positions_y.resize(n);
        positions_z.resize(n);
        momenta_x.resize(n);
        momenta_y.resize(n);
        momenta_z.resize(n);
        if (reuse_accelerations) {
            accelerations_x.resize(n);
            accelerations_y.resize(n);
            accelerations_z.resize(n);
        }
        if (!uniform_mass.has_value()) masses.resize(n);
        ids.resize(n);

        num_owned_particles_ = n;
        accel_validity = FieldValidity::INVALID;
        ghost_validity = FieldValidity::INVALID;
    }

    // Declares that all currently stored sources are owned. This is legal only
    // when no compact ghost suffix is present, because ghosts intentionally omit
    // momentum. Ghost exchange normally calls clear_ghosts() before this method.
    void mark_ghost_start() {
        validate_storage_layout();
        if (num_owned_particles_ != size()) {
            throw std::logic_error(
                "Cannot mark a ghost start while a previous ghost suffix exists");
        }
        num_owned_particles_ = positions_x.size();
        ghost_validity = FieldValidity::INVALID;
    }

    void clear_ghosts() {
        validate_storage_layout();
        if (num_owned_particles_ < positions_x.size()) {
            resize_internal(num_owned_particles_);
            if (accel_validity == FieldValidity::VALID) {
                accel_validity = FieldValidity::STALE;
            }
        }
        ghost_validity = FieldValidity::INVALID;
    }

    // Reserve only source fields required by TreePM. Momentum and acceleration
    // are owned integration state and deliberately do not scale with ghost count.
    void reserve_ghost_capacity(std::size_t ghost_count) {
        validate_storage_layout();
        if (ghost_count
            > std::numeric_limits<std::size_t>::max() - num_owned_particles_) {
            throw std::length_error("ParticleStore ghost capacity overflows size_t");
        }
        const std::size_t target_size = num_owned_particles_ + ghost_count;
        positions_x.reserve(target_size);
        positions_y.reserve(target_size);
        positions_z.reserve(target_size);
        if (!uniform_mass.has_value()) masses.reserve(target_size);
        ids.reserve(target_size);
    }

    void append_owned_particle(Position pos,
                               Momentum mom,
                               Acceleration acc,
                               Real mass,
                               ParticleId id) {
        validate_storage_layout();
        if (num_owned_particles_ != size()) {
            throw std::logic_error(
                "Cannot append an owned particle while ghost particles are present");
        }
        append_owned_fields(pos, mom, acc, mass, id);
        ++num_owned_particles_;
        ghost_validity = FieldValidity::INVALID;
    }

    void append_ghost_particle(Position pos,
                               Real mass,
                               ParticleId id) {
        validate_storage_layout();
        if (num_owned_particles_ > size()) {
            throw std::logic_error("ParticleStore owned count exceeds storage size");
        }
        if (!std::isfinite(pos.vec.x)
            || !std::isfinite(pos.vec.y)
            || !std::isfinite(pos.vec.z)
            || !std::isfinite(mass)
            || mass <= 0.0) {
            throw std::invalid_argument(
                "Appended ghost position and mass must be finite; mass must be positive");
        }
        if (uniform_mass.has_value() && mass != *uniform_mass) {
            throw std::invalid_argument(
                "Appended ghost mass does not match the uniform-mass representation");
        }
        if (size() == std::numeric_limits<std::size_t>::max()) {
            throw std::length_error("ParticleStore size cannot be incremented");
        }

        const std::size_t target_size = size() + 1;
        positions_x.reserve(target_size);
        positions_y.reserve(target_size);
        positions_z.reserve(target_size);
        if (!uniform_mass.has_value()) masses.reserve(target_size);
        ids.reserve(target_size);

        positions_x.push_back(pos.vec.x);
        positions_y.push_back(pos.vec.y);
        positions_z.push_back(pos.vec.z);
        ids.push_back(id);
        if (!uniform_mass.has_value()) masses.push_back(mass);
        if (accel_validity == FieldValidity::VALID) {
            // Admitting a new force source changes the force represented by an
            // otherwise-valid owned acceleration field.
            accel_validity = FieldValidity::STALE;
        }
        ghost_validity = FieldValidity::VALID;
    }

    void remove_particle(std::size_t i) {
        if (i >= size()) {
            throw std::out_of_range("Particle removal index out of range");
        }
        if (num_owned_particles_ != size()) {
            throw std::logic_error(
                "Cannot swap-and-pop owned particles while ghosts are present");
        }
        validate_storage_layout();

        const std::size_t last = size() - 1;
        if (i != last) {
            positions_x[i] = positions_x[last];
            positions_y[i] = positions_y[last];
            positions_z[i] = positions_z[last];
            momenta_x[i] = momenta_x[last];
            momenta_y[i] = momenta_y[last];
            momenta_z[i] = momenta_z[last];
            if (acceleration_storage_enabled_) {
                accelerations_x[i] = accelerations_x[last];
                accelerations_y[i] = accelerations_y[last];
                accelerations_z[i] = accelerations_z[last];
            }
            if (!uniform_mass.has_value()) masses[i] = masses[last];
            ids[i] = ids[last];
        }
        resize_internal(last);
        num_owned_particles_ = last;
        accel_validity = FieldValidity::INVALID;
        ghost_validity = FieldValidity::INVALID;
    }

private:
    void validate_storage_layout() const {
        const std::size_t total = positions_x.size();
        const bool source_layout_valid = positions_y.size() == total
            && positions_z.size() == total
            && ids.size() == total
            && (uniform_mass.has_value() || masses.size() == total);
        if (!source_layout_valid || num_owned_particles_ > total) {
            throw std::logic_error(
                "ParticleStore source component sizes or owned count are inconsistent");
        }
        if (momenta_x.size() != num_owned_particles_
            || momenta_y.size() != num_owned_particles_
            || momenta_z.size() != num_owned_particles_) {
            throw std::logic_error(
                "ParticleStore owned momentum component sizes are inconsistent");
        }

        const bool all_empty = accelerations_x.empty()
            && accelerations_y.empty()
            && accelerations_z.empty();
        const bool all_owned =
            accelerations_x.size() == num_owned_particles_
            && accelerations_y.size() == num_owned_particles_
            && accelerations_z.size() == num_owned_particles_;
        if (acceleration_storage_enabled_) {
            if (!all_owned) {
                throw std::logic_error(
                    "ParticleStore owned acceleration component sizes are inconsistent");
            }
        } else if (!all_empty) {
            throw std::logic_error(
                "ParticleStore omitted acceleration state retains component storage");
        }
    }

    void require_valid_acceleration_storage() const {
        validate_storage_layout();
        if (accel_validity != FieldValidity::VALID) {
            throw std::logic_error("Accelerations are not valid");
        }
        if (!acceleration_storage_enabled_) {
            throw std::logic_error(
                "Valid acceleration state has no component storage");
        }
    }

    void append_owned_fields(Position pos,
                             Momentum mom,
                             Acceleration acc,
                             Real mass,
                             ParticleId id) {
        validate_storage_layout();
        if (!std::isfinite(pos.vec.x)
            || !std::isfinite(pos.vec.y)
            || !std::isfinite(pos.vec.z)
            || !std::isfinite(mom.vec.x)
            || !std::isfinite(mom.vec.y)
            || !std::isfinite(mom.vec.z)
            || !std::isfinite(mass)
            || mass <= 0.0) {
            throw std::invalid_argument(
                "Appended particle position, momentum, and mass must be finite; mass must be positive");
        }
        if (acceleration_storage_enabled_
            && (!std::isfinite(acc.vec.x)
                || !std::isfinite(acc.vec.y)
                || !std::isfinite(acc.vec.z))) {
            throw std::invalid_argument(
                "Appended particle acceleration must be finite when stored");
        }
        if (uniform_mass.has_value() && mass != *uniform_mass) {
            throw std::invalid_argument(
                "Appended particle mass does not match the uniform-mass representation");
        }
        if (size() == std::numeric_limits<std::size_t>::max()) {
            throw std::length_error("ParticleStore size cannot be incremented");
        }

        // Reserve every participating owned/source SoA component before
        // publishing any new element. A failed allocation may change capacity,
        // but cannot leave fields with different logical lengths.
        const std::size_t target_size = size() + 1;
        positions_x.reserve(target_size);
        positions_y.reserve(target_size);
        positions_z.reserve(target_size);
        momenta_x.reserve(target_size);
        momenta_y.reserve(target_size);
        momenta_z.reserve(target_size);
        if (acceleration_storage_enabled_) {
            accelerations_x.reserve(target_size);
            accelerations_y.reserve(target_size);
            accelerations_z.reserve(target_size);
        }
        if (!uniform_mass.has_value()) masses.reserve(target_size);
        ids.reserve(target_size);

        positions_x.push_back(pos.vec.x);
        positions_y.push_back(pos.vec.y);
        positions_z.push_back(pos.vec.z);
        momenta_x.push_back(mom.vec.x);
        momenta_y.push_back(mom.vec.y);
        momenta_z.push_back(mom.vec.z);
        if (acceleration_storage_enabled_) {
            accelerations_x.push_back(acc.vec.x);
            accelerations_y.push_back(acc.vec.y);
            accelerations_z.push_back(acc.vec.z);
        }
        ids.push_back(id);
        if (!uniform_mass.has_value()) masses.push_back(mass);
    }

    void resize_internal(std::size_t n) {
        validate_storage_layout();
        if (n > size() || n > num_owned_particles_) {
            throw std::logic_error(
                "ParticleStore internal resize may only shrink to an owned prefix");
        }
        positions_x.resize(n);
        positions_y.resize(n);
        positions_z.resize(n);
        momenta_x.resize(n);
        momenta_y.resize(n);
        momenta_z.resize(n);
        if (acceleration_storage_enabled_) {
            accelerations_x.resize(n);
            accelerations_y.resize(n);
            accelerations_z.resize(n);
        }
        if (!uniform_mass.has_value()) masses.resize(n);
        ids.resize(n);
    }

    // Source fields: owned prefix plus compact read-only ghost suffix.
    std::vector<Real> positions_x;
    std::vector<Real> positions_y;
    std::vector<Real> positions_z;
    std::vector<Real> masses;
    std::vector<ParticleId> ids;

    // Integration fields: owned prefix only; ghosts never carry these values.
    std::vector<Real> momenta_x;
    std::vector<Real> momenta_y;
    std::vector<Real> momenta_z;
    std::vector<Real> accelerations_x;
    std::vector<Real> accelerations_y;
    std::vector<Real> accelerations_z;

    FieldValidity accel_validity{FieldValidity::INVALID};
    FieldValidity ghost_validity{FieldValidity::INVALID};
    std::optional<Real> uniform_mass;
    std::string verified_snapshot_ic_sha256;

    std::size_t num_owned_particles_{0};
    bool acceleration_storage_enabled_{true};
};

} // namespace core
} // namespace cosmo_nbody
