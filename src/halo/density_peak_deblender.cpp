#include "cosmo_nbody/halo/density_peak_deblender.hpp"

#include "cosmo_nbody/math/exact_positive_sum.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody::halo {
namespace {

constexpr std::size_t invalid_index =
    std::numeric_limits<std::size_t>::max();
constexpr std::size_t merge_query_batch_target_bytes =
    std::size_t{4} * 1024 * 1024;
constexpr std::size_t minimum_parallel_neighbor_queries = 256;

struct LocalParticle {
    std::size_t store_index{0};
    core::ParticleId id{0};
    core::Vec3 position{};
    core::Real mass{0.0};
};

struct Neighbor {
    std::size_t local_index{0};
    core::ParticleId id{0};
    long double distance{0.0L};
};

struct KdNode {
    std::size_t particle_index{invalid_index};
    std::size_t left{invalid_index};
    std::size_t right{invalid_index};
    core::Vec3 lower{};
    core::Vec3 upper{};
};

std::size_t checked_add(
    std::size_t lhs,
    std::size_t rhs,
    const char* context) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(context);
    }
    return lhs + rhs;
}

std::size_t checked_multiply(
    std::size_t lhs,
    std::size_t rhs,
    const char* context) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(context);
    }
    return lhs * rhs;
}

template <typename T>
void add_array_bytes(
    std::size_t& total,
    std::size_t count,
    const char* context) {
    total = checked_add(
        total,
        checked_multiply(count, sizeof(T), context),
        context);
}

bool neighbor_is_closer(const Neighbor& lhs, const Neighbor& rhs) {
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.id < rhs.id;
}

long double periodic_distance(
    const core::Vec3& first,
    const core::Vec3& second,
    core::Real box_size) {
    const core::Vec3 displacement = math::minimum_image_displacement(
        first, second, box_size);
    const long double distance = core::scale_safe_norm3(
        static_cast<long double>(displacement.x),
        static_cast<long double>(displacement.y),
        static_cast<long double>(displacement.z));
    if (!std::isfinite(distance) || distance < 0.0L) {
        throw std::invalid_argument(
            "Density peak deblending encountered invalid periodic geometry");
    }
    return distance;
}

core::Real coordinate(const core::Vec3& point, int axis) {
    if (axis == 0) return point.x;
    if (axis == 1) return point.y;
    return point.z;
}

void extend_bounds(
    core::Vec3& lower,
    core::Vec3& upper,
    const core::Vec3& point) {
    lower.x = std::min(lower.x, point.x);
    lower.y = std::min(lower.y, point.y);
    lower.z = std::min(lower.z, point.z);
    upper.x = std::max(upper.x, point.x);
    upper.y = std::max(upper.y, point.y);
    upper.z = std::max(upper.z, point.z);
}

bool is_canonical_position(
    const core::Vec3& position,
    core::Real box_size) {
    return position.x >= 0.0 && position.x < box_size
        && position.y >= 0.0 && position.y < box_size
        && position.z >= 0.0 && position.z < box_size;
}

core::Real positive_difference_down(
    core::Real high,
    core::Real low) {
    if (!std::isfinite(high) || !std::isfinite(low)) return 0.0;
    const core::Real difference = high - low;
    if (!(difference > 0.0) || !std::isfinite(difference)) return 0.0;
    return std::nextafter(difference, core::Real{0.0});
}

core::Real interval_distance_down(
    core::Real query,
    core::Real lower,
    core::Real upper) {
    if (!std::isfinite(query) || !std::isfinite(lower)
        || !std::isfinite(upper) || lower > upper) {
        return 0.0;
    }
    if (query < lower) return positive_difference_down(lower, query);
    if (query > upper) return positive_difference_down(query, upper);
    return 0.0;
}

core::Real periodic_interval_distance_down(
    core::Real query,
    core::Real lower,
    core::Real upper,
    core::Real box_size) {
    const core::Real lower_minus_box = lower - box_size;
    const core::Real upper_minus_box = upper - box_size;
    const core::Real lower_plus_box = lower + box_size;
    const core::Real upper_plus_box = upper + box_size;
    if (!std::isfinite(lower_minus_box)
        || !std::isfinite(upper_minus_box)
        || !std::isfinite(lower_plus_box)
        || !std::isfinite(upper_plus_box)) {
        return 0.0;
    }
    return std::min({
        interval_distance_down(query, lower, upper),
        interval_distance_down(query, lower_minus_box, upper_minus_box),
        interval_distance_down(query, lower_plus_box, upper_plus_box),
    });
}

long double periodic_box_distance_down(
    const core::Vec3& query,
    const core::Vec3& lower,
    const core::Vec3& upper,
    core::Real box_size) {
    const core::Real x = periodic_interval_distance_down(
        query.x, lower.x, upper.x, box_size);
    const core::Real y = periodic_interval_distance_down(
        query.y, lower.y, upper.y, box_size);
    const core::Real z = periodic_interval_distance_down(
        query.z, lower.z, upper.z, box_size);
    const long double distance = core::scale_safe_norm3(
        static_cast<long double>(x),
        static_cast<long double>(y),
        static_cast<long double>(z));
    if (!(distance > 0.0L) || !std::isfinite(distance)) return 0.0L;
    return std::nextafter(distance, 0.0L);
}

class PeriodicKdTree {
public:
    PeriodicKdTree(
        const std::vector<LocalParticle>& local,
        core::Real box_size)
        : local_(local), box_size_(box_size) {
        build_order_.resize(local_.size());
        std::iota(
            build_order_.begin(), build_order_.end(), std::size_t{0});
        nodes_.reserve(local_.size());
        root_ = build(0, build_order_.size());
        if (nodes_.size() != local_.size() || root_ == invalid_index) {
            throw std::logic_error(
                "Density peak periodic k-d tree construction lost particles");
        }
        std::vector<std::size_t>{}.swap(build_order_);
    }

    void select_exact_neighbors(
        std::size_t center_index,
        std::size_t effective_k,
        std::vector<Neighbor>& selection_scratch) const {
        if (center_index >= local_.size() || effective_k == 0
            || effective_k >= local_.size()) {
            throw std::logic_error(
                "Density peak periodic k-d tree query received an invalid extent");
        }
        selection_scratch.clear();
        query_node(root_, center_index, effective_k, selection_scratch);
        if (selection_scratch.size() != effective_k) {
            throw std::logic_error(
                "Density peak periodic k-d tree query returned too few unique neighbors");
        }
        std::sort(
            selection_scratch.begin(),
            selection_scratch.end(),
            neighbor_is_closer);
    }

private:
    std::size_t build(std::size_t begin, std::size_t end) {
        if (begin == end) return invalid_index;

        core::Vec3 lower = local_[build_order_[begin]].position;
        core::Vec3 upper = lower;
        for (std::size_t index = begin + 1; index < end; ++index) {
            extend_bounds(
                lower, upper, local_[build_order_[index]].position);
        }
        const core::Real spans[3]{
            upper.x - lower.x,
            upper.y - lower.y,
            upper.z - lower.z,
        };
        int axis = 0;
        if (spans[1] > spans[axis]) axis = 1;
        if (spans[2] > spans[axis]) axis = 2;

        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(
            build_order_.begin() + static_cast<std::ptrdiff_t>(begin),
            build_order_.begin() + static_cast<std::ptrdiff_t>(middle),
            build_order_.begin() + static_cast<std::ptrdiff_t>(end),
            [&](std::size_t lhs, std::size_t rhs) {
                const core::Real lhs_coordinate = coordinate(
                    local_[lhs].position, axis);
                const core::Real rhs_coordinate = coordinate(
                    local_[rhs].position, axis);
                if (lhs_coordinate != rhs_coordinate) {
                    return lhs_coordinate < rhs_coordinate;
                }
                return local_[lhs].id < local_[rhs].id;
            });

        const std::size_t node_index = nodes_.size();
        nodes_.push_back({});
        const std::size_t left = build(begin, middle);
        const std::size_t right = build(middle + 1, end);

        KdNode node;
        node.particle_index = build_order_[middle];
        node.left = left;
        node.right = right;
        node.lower = local_[node.particle_index].position;
        node.upper = node.lower;
        if (left != invalid_index) {
            extend_bounds(node.lower, node.upper, nodes_[left].lower);
            extend_bounds(node.lower, node.upper, nodes_[left].upper);
        }
        if (right != invalid_index) {
            extend_bounds(node.lower, node.upper, nodes_[right].lower);
            extend_bounds(node.lower, node.upper, nodes_[right].upper);
        }
        nodes_[node_index] = node;
        return node_index;
    }

    void consider_neighbor(
        std::size_t candidate_index,
        std::size_t center_index,
        std::size_t effective_k,
        std::vector<Neighbor>& selection_scratch) const {
        if (candidate_index == center_index) return;
        const Neighbor candidate{
            candidate_index,
            local_[candidate_index].id,
            periodic_distance(
                local_[center_index].position,
                local_[candidate_index].position,
                box_size_),
        };
        if (selection_scratch.size() < effective_k) {
            selection_scratch.push_back(candidate);
            std::push_heap(
                selection_scratch.begin(),
                selection_scratch.end(),
                neighbor_is_closer);
            return;
        }
        if (!neighbor_is_closer(candidate, selection_scratch.front())) return;
        std::pop_heap(
            selection_scratch.begin(),
            selection_scratch.end(),
            neighbor_is_closer);
        selection_scratch.back() = candidate;
        std::push_heap(
            selection_scratch.begin(),
            selection_scratch.end(),
            neighbor_is_closer);
    }

    void query_node(
        std::size_t node_index,
        std::size_t center_index,
        std::size_t effective_k,
        std::vector<Neighbor>& selection_scratch) const {
        if (node_index == invalid_index) return;
        const KdNode& node = nodes_[node_index];
        if (selection_scratch.size() == effective_k
            && periodic_box_distance_down(
                   local_[center_index].position,
                   node.lower,
                   node.upper,
                   box_size_)
                > selection_scratch.front().distance) {
            return;
        }

        consider_neighbor(
            node.particle_index,
            center_index,
            effective_k,
            selection_scratch);

        std::size_t first = node.left;
        std::size_t second = node.right;
        if (first != invalid_index && second != invalid_index) {
            const long double first_distance =
                periodic_box_distance_down(
                    local_[center_index].position,
                    nodes_[first].lower,
                    nodes_[first].upper,
                    box_size_);
            const long double second_distance =
                periodic_box_distance_down(
                    local_[center_index].position,
                    nodes_[second].lower,
                    nodes_[second].upper,
                    box_size_);
            if (second_distance < first_distance
                || (second_distance == first_distance && second < first)) {
                std::swap(first, second);
            }
        }
        query_node(
            first,
            center_index,
            effective_k,
            selection_scratch);
        query_node(
            second,
            center_index,
            effective_k,
            selection_scratch);
    }

    const std::vector<LocalParticle>& local_;
    core::Real box_size_{0.0};
    std::vector<KdNode> nodes_;
    std::vector<std::size_t> build_order_;
    std::size_t root_{invalid_index};
};

void select_exact_neighbors_bruteforce(
    const std::vector<LocalParticle>& local,
    std::size_t center_index,
    std::size_t effective_k,
    core::Real box_size,
    std::vector<Neighbor>& selection_scratch) {
    if (center_index >= local.size() || effective_k == 0
        || effective_k >= local.size()) {
        throw std::logic_error(
            "Density peak exact-neighbor selection received an invalid extent");
    }
    selection_scratch.clear();
    for (std::size_t candidate_index = 0;
         candidate_index < local.size();
         ++candidate_index) {
        if (candidate_index == center_index) continue;
        selection_scratch.push_back({
            candidate_index,
            local[candidate_index].id,
            periodic_distance(
                local[center_index].position,
                local[candidate_index].position,
                box_size),
        });
    }
    if (effective_k < selection_scratch.size()) {
        std::nth_element(
            selection_scratch.begin(),
            selection_scratch.begin()
                + static_cast<std::ptrdiff_t>(effective_k),
            selection_scratch.end(),
            neighbor_is_closer);
        selection_scratch.resize(effective_k);
    }
    std::sort(
        selection_scratch.begin(),
        selection_scratch.end(),
        neighbor_is_closer);
}

void select_exact_neighbors(
    const std::vector<LocalParticle>& local,
    const PeriodicKdTree* tree,
    std::size_t center_index,
    std::size_t effective_k,
    core::Real box_size,
    std::vector<Neighbor>& selection_scratch) {
    if (tree != nullptr) {
        tree->select_exact_neighbors(
            center_index, effective_k, selection_scratch);
        return;
    }
    select_exact_neighbors_bruteforce(
        local,
        center_index,
        effective_k,
        box_size,
        selection_scratch);
}

std::size_t deblend_worker_count() {
#ifdef COSMO_NBODY_HAS_OPENMP
    const int maximum_threads = omp_get_max_threads();
    if (maximum_threads > 0) {
        return static_cast<std::size_t>(maximum_threads);
    }
#endif
    return 1;
}

std::size_t merge_query_batch_capacity(
    std::size_t particle_count,
    std::size_t effective_k) {
    if (particle_count == 0 || effective_k == 0) return 0;
    const std::size_t bytes_per_particle = checked_add(
        checked_multiply(
            effective_k,
            sizeof(std::size_t),
            "Density peak merge batch row bytes overflow size_t"),
        sizeof(std::size_t),
        "Density peak merge batch row bytes overflow size_t");
    return std::min(
        particle_count,
        std::max<std::size_t>(
            1,
            merge_query_batch_target_bytes / bytes_per_particle));
}

template <typename Function>
void for_each_exact_neighbor_set(
    const std::vector<LocalParticle>& local,
    const PeriodicKdTree* tree,
    std::size_t begin,
    std::size_t count,
    std::size_t effective_k,
    core::Real box_size,
    Function&& function) {
    if (count == 0) return;
    if (begin > local.size() || count > local.size() - begin) {
        throw std::logic_error(
            "Density peak exact-neighbor pass exceeds the local particle range");
    }

    const std::size_t available_workers = deblend_worker_count();
    if (tree == nullptr || available_workers == 1
        || count < minimum_parallel_neighbor_queries) {
        std::vector<Neighbor> selection_scratch;
        selection_scratch.reserve(
            tree == nullptr ? local.size() - 1 : effective_k);
        for (std::size_t offset = 0; offset < count; ++offset) {
            const std::size_t center_index = begin + offset;
            select_exact_neighbors(
                local,
                tree,
                center_index,
                effective_k,
                box_size,
                selection_scratch);
            function(center_index, selection_scratch);
        }
        return;
    }

#ifdef COSMO_NBODY_HAS_OPENMP
    const int worker_count = static_cast<int>(std::min(
        count,
        std::min(
            available_workers,
            static_cast<std::size_t>(std::numeric_limits<int>::max()))));
    std::atomic<bool> worker_failed{false};
    std::exception_ptr worker_exception;
    #pragma omp parallel num_threads(worker_count)
    {
        std::vector<Neighbor> selection_scratch;
        selection_scratch.reserve(effective_k);
        #pragma omp for schedule(dynamic, 8)
        for (std::size_t offset = 0; offset < count; ++offset) {
            if (worker_failed.load(std::memory_order_relaxed)) continue;
            try {
                const std::size_t center_index = begin + offset;
                select_exact_neighbors(
                    local,
                    tree,
                    center_index,
                    effective_k,
                    box_size,
                    selection_scratch);
                function(center_index, selection_scratch);
            } catch (...) {
                worker_failed.store(true, std::memory_order_relaxed);
                #pragma omp critical(cosmo_nbody_deblend_worker_failure)
                {
                    if (!worker_exception) {
                        worker_exception = std::current_exception();
                    }
                }
            }
        }
    }
    if (worker_exception) std::rethrow_exception(worker_exception);
#else
    throw std::logic_error(
        "Density peak parallel path was selected without OpenMP support");
#endif
}

std::size_t find_root(
    std::vector<std::size_t>& parents,
    std::size_t value) {
    std::size_t root = value;
    while (parents[root] != root) root = parents[root];
    while (parents[value] != value) {
        const std::size_t next = parents[value];
        parents[value] = root;
        value = next;
    }
    return root;
}

bool peak_is_stronger(
    std::size_t first,
    std::size_t second,
    const std::vector<core::Real>& densities,
    const std::vector<LocalParticle>& local) {
    if (densities[first] != densities[second]) {
        return densities[first] > densities[second];
    }
    return local[first].id < local[second].id;
}

std::size_t unite_peak_roots(
    std::vector<std::size_t>& parents,
    std::size_t first,
    std::size_t second,
    const std::vector<core::Real>& densities,
    const std::vector<LocalParticle>& local) {
    first = find_root(parents, first);
    second = find_root(parents, second);
    if (first == second) return first;
    const std::size_t stronger = peak_is_stronger(
        first, second, densities, local) ? first : second;
    const std::size_t weaker = stronger == first ? second : first;
    parents[weaker] = stronger;
    return stronger;
}

} // namespace

DensityPeakDeblender::DensityPeakDeblender(
    DensityPeakDeblendOptions options)
    : options_(std::move(options)) {
    if (options_.k_neighbors == 0) {
        throw std::invalid_argument(
            "Density peak deblending requires k_neighbors > 0");
    }
    if (options_.minimum_host_particles == 0) {
        throw std::invalid_argument(
            "Density peak minimum_host_particles must be positive");
    }
    if (!std::isfinite(options_.saddle_to_lower_peak_merge_ratio)
        || options_.saddle_to_lower_peak_merge_ratio < 0.0
        || options_.saddle_to_lower_peak_merge_ratio > 1.0) {
        throw std::invalid_argument(
            "Density peak saddle merge ratio must lie in [0,1]");
    }
}

DensityPeakDeblendExecutionPlan DensityPeakDeblender::execution_plan(
    std::size_t candidate_particle_count) const {
    if (candidate_particle_count < 2) {
        throw std::invalid_argument(
            "Density peak execution plan requires at least two particles");
    }
    DensityPeakDeblendExecutionPlan plan;
    plan.particle_count = candidate_particle_count;
    plan.effective_k_neighbors = std::min(
        options_.k_neighbors, candidate_particle_count - 1);
    plan.maximum_directed_knn_edges = checked_multiply(
        candidate_particle_count,
        plan.effective_k_neighbors,
        "Density peak maximum directed kNN edge count overflows size_t");

    std::size_t bytes = 0;
    add_array_bytes<LocalParticle>(
        bytes, candidate_particle_count,
        "Density peak local-particle bytes overflow size_t");
    add_array_bytes<std::size_t>(
        bytes, candidate_particle_count,
        "Density peak duplicate-index bytes overflow size_t");
    add_array_bytes<KdNode>(
        bytes, candidate_particle_count,
        "Density peak periodic k-d tree bytes overflow size_t");
    add_array_bytes<std::size_t>(
        bytes, candidate_particle_count,
        "Density peak periodic k-d tree build-order bytes overflow size_t");
    add_array_bytes<Neighbor>(
        bytes,
        checked_multiply(
            deblend_worker_count(),
            plan.effective_k_neighbors,
            "Density peak worker scratch count overflows size_t"),
        "Density peak worker scratch bytes overflow size_t");
    const std::size_t merge_batch_capacity = merge_query_batch_capacity(
        candidate_particle_count, plan.effective_k_neighbors);
    add_array_bytes<std::size_t>(
        bytes,
        checked_multiply(
            merge_batch_capacity,
            plan.effective_k_neighbors,
            "Density peak merge batch edge count overflows size_t"),
        "Density peak merge batch edge bytes overflow size_t");
    add_array_bytes<std::size_t>(
        bytes, merge_batch_capacity,
        "Density peak merge batch count bytes overflow size_t");
    add_array_bytes<core::Real>(
        bytes, candidate_particle_count,
        "Density peak density bytes overflow size_t");
    // ascent_parent, initial_root, peak_roots, merge_parent, retained_roots
    add_array_bytes<std::size_t>(
        bytes,
        checked_multiply(
            candidate_particle_count, std::size_t{5},
            "Density peak linear index count overflows size_t"),
        "Density peak linear index bytes overflow size_t");
    add_array_bytes<std::vector<std::size_t>>(
        bytes, candidate_particle_count,
        "Density peak final-members outer bytes overflow size_t");
    // Final-members payload and result-host membership payload are included
    // simultaneously to remain conservative across ownership transfer.
    add_array_bytes<std::size_t>(
        bytes,
        checked_multiply(
            candidate_particle_count, std::size_t{2},
            "Density peak membership index count overflows size_t"),
        "Density peak membership index bytes overflow size_t");
    add_array_bytes<DensityPeakRecord>(
        bytes, candidate_particle_count,
        "Density peak result-record bytes overflow size_t");
    add_array_bytes<DeblendedHostSeed>(
        bytes, candidate_particle_count,
        "Density peak host-record bytes overflow size_t");
    plan.conservative_payload_peak_bytes = bytes;
    return plan;
}

DensityPeakDeblendResult DensityPeakDeblender::deblend(
    const core::ParticleStore& particles,
    const FoFMembership& candidate,
    core::Real box_size) const {
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "Density peak deblending requires a finite positive box size");
    }
    if (candidate.particle_indices.size() < 2) {
        throw std::invalid_argument(
            "Density peak deblending requires at least two candidate particles");
    }
    const DensityPeakDeblendExecutionPlan plan = execution_plan(
        candidate.particle_indices.size());

    const auto position_x = particles.get_positions_x();
    const auto position_y = particles.get_positions_y();
    const auto position_z = particles.get_positions_z();
    const auto particle_ids = particles.get_ids();

    std::vector<std::size_t> sorted_store_indices = candidate.particle_indices;
    std::sort(sorted_store_indices.begin(), sorted_store_indices.end());
    if (std::adjacent_find(
            sorted_store_indices.begin(), sorted_store_indices.end())
        != sorted_store_indices.end()) {
        throw std::invalid_argument(
            "Density peak candidate contains a duplicate particle index");
    }

    std::vector<LocalParticle> local;
    local.reserve(candidate.particle_indices.size());
    bool all_positions_are_canonical = true;
    for (const std::size_t index : candidate.particle_indices) {
        if (index >= particles.num_owned_particles()) {
            throw std::out_of_range(
                "Density peak candidate contains a non-owned particle index");
        }
        const core::ParticleId id = particle_ids[index];
        const core::Vec3 position{
            position_x[index], position_y[index], position_z[index]};
        if (!std::isfinite(position.x) || !std::isfinite(position.y)
            || !std::isfinite(position.z)) {
            throw std::invalid_argument(
                "Density peak candidate contains a non-finite position");
        }
        all_positions_are_canonical = all_positions_are_canonical
            && is_canonical_position(position, box_size);
        const core::Real mass = particles.mass_at(index);
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::invalid_argument(
                "Density peak candidate contains a non-positive particle mass");
        }
        local.push_back({index, id, position, mass});
    }
    std::sort(
        local.begin(), local.end(), [](const LocalParticle& lhs, const LocalParticle& rhs) {
            return lhs.id < rhs.id;
        });
    if (std::adjacent_find(
            local.begin(), local.end(),
            [](const LocalParticle& lhs, const LocalParticle& rhs) {
                return lhs.id == rhs.id;
            })
        != local.end()) {
        throw std::invalid_argument(
            "Density peak candidate contains duplicate stable particle IDs");
    }
    std::vector<std::size_t>{}.swap(sorted_store_indices);

    const std::size_t count = local.size();
    const std::size_t effective_k = plan.effective_k_neighbors;
    std::unique_ptr<PeriodicKdTree> tree;
    if (all_positions_are_canonical) {
        tree = std::make_unique<PeriodicKdTree>(local, box_size);
    }
    std::vector<core::Real> densities(count, 0.0);

    // Recompute the exact deterministic kNN set for density estimation. The
    // balanced periodic k-d tree changes only candidate discovery; final distance
    // and ParticleID tie-breaking remain the brute-force comparison rules.
    for_each_exact_neighbor_set(
        local,
        tree.get(),
        0,
        count,
        effective_k,
        box_size,
        [&](std::size_t i, const std::vector<Neighbor>& neighbors) {
            const long double radius = neighbors.back().distance;
            if (!(radius > 0.0L) || !std::isfinite(radius)) {
                throw std::invalid_argument(
                    "Density peak k-neighbour radius is singular; coincident particle positions make the estimator undefined");
            }
            math::ExactPositiveDoubleSum enclosed_mass_accumulator;
            enclosed_mass_accumulator.add(local[i].mass);
            for (const Neighbor& neighbor : neighbors) {
                if (neighbor.local_index >= local.size()) {
                    throw std::logic_error(
                        "Density peak neighbor index is outside the candidate");
                }
                enclosed_mass_accumulator.add(
                    local[neighbor.local_index].mass);
            }
            // Enclosed mass is an intermediate. Two finite particle masses near
            // the binary64 ceiling can overflow binary64 while mass / r^3 remains
            // representable; take the logarithm of the exact sum instead.
            const long double log_enclosed_mass =
                enclosed_mass_accumulator.log_value();
            if (!std::isfinite(log_enclosed_mass)) {
                throw std::overflow_error(
                    "Density peak enclosed mass logarithm is not finite");
            }
            const long double log_density =
                log_enclosed_mass
                - std::log((4.0L / 3.0L) * std::numbers::pi_v<long double>)
                - 3.0L * std::log(radius);
            const long double minimum_log_density = std::log(
                static_cast<long double>(
                    std::numeric_limits<core::Real>::denorm_min()));
            const long double maximum_log_density = std::log(
                static_cast<long double>(
                    std::numeric_limits<core::Real>::max()));
            if (!std::isfinite(log_density)
                || log_density < minimum_log_density
                || log_density > maximum_log_density) {
                throw std::overflow_error(
                    "Density peak estimate is not representable");
            }
            const long double density = std::exp(log_density);
            if (!(density > 0.0L) || !std::isfinite(density)) {
                throw std::overflow_error(
                    "Density peak estimate is not finite and positive");
            }
            densities[i] = static_cast<core::Real>(density);
        });

    std::vector<std::size_t> ascent_parent(count);
    std::iota(ascent_parent.begin(), ascent_parent.end(), std::size_t{0});

    // Recompute the same exact kNN set for discrete density ascent. Independent
    // particle queries are parallel; each writes only its own deterministic row.
    for_each_exact_neighbor_set(
        local,
        tree.get(),
        0,
        count,
        effective_k,
        box_size,
        [&](std::size_t i, const std::vector<Neighbor>& neighbors) {
            std::size_t best = i;
            for (const Neighbor& neighbor : neighbors) {
                const std::size_t j = neighbor.local_index;
                if (densities[j] > densities[best]
                    || (densities[j] == densities[best]
                        && local[j].id < local[best].id)) {
                    best = j;
                }
            }
            if (densities[best] > densities[i]
                || (densities[best] == densities[i]
                    && local[best].id < local[i].id)) {
                ascent_parent[i] = best;
            }
        });
    for (std::size_t i = 0; i < count; ++i) {
        (void)find_root(ascent_parent, i);
    }

    std::vector<std::size_t> initial_root(count);
    std::vector<std::size_t> peak_roots;
    peak_roots.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        initial_root[i] = find_root(ascent_parent, i);
        if (initial_root[i] == i) peak_roots.push_back(i);
    }
    if (peak_roots.empty()) {
        throw std::logic_error("Density ascent produced no peak roots");
    }

    std::vector<std::size_t> merge_parent(count);
    std::iota(merge_parent.begin(), merge_parent.end(), std::size_t{0});
    std::size_t observed_cross_basin_edges = 0;

    // Query exact neighbors in bounded parallel batches, then replay qualifying
    // directed edges serially in deterministic (particle, neighbor-rank) order.
    // This preserves the strongest-density/lowest-ParticleID representative
    // exactly while avoiding both O(N^2) search and retained O(N*k) topology
    // storage.
    const std::size_t merge_batch_capacity = merge_query_batch_capacity(
        count, effective_k);
    std::vector<std::size_t> qualifying_neighbors(
        checked_multiply(
            merge_batch_capacity,
            effective_k,
            "Density peak merge batch edge count overflows size_t"),
        invalid_index);
    std::vector<std::size_t> cross_basin_counts(
        merge_batch_capacity, std::size_t{0});
    for (std::size_t batch_begin = 0;
         batch_begin < count;
         batch_begin += merge_batch_capacity) {
        const std::size_t batch_count = std::min(
            merge_batch_capacity, count - batch_begin);
        for_each_exact_neighbor_set(
            local,
            tree.get(),
            batch_begin,
            batch_count,
            effective_k,
            box_size,
            [&](std::size_t i, const std::vector<Neighbor>& neighbors) {
                const std::size_t row = i - batch_begin;
                cross_basin_counts[row] = 0;
                const std::size_t row_begin = row * effective_k;
                if (neighbors.size() != effective_k) {
                    throw std::logic_error(
                        "Density peak merge query returned a truncated neighbor set");
                }
                for (std::size_t neighbor_rank = 0;
                     neighbor_rank < effective_k;
                     ++neighbor_rank) {
                    qualifying_neighbors[row_begin + neighbor_rank] =
                        invalid_index;
                    const std::size_t j =
                        neighbors[neighbor_rank].local_index;
                    if (j >= local.size()) {
                        throw std::logic_error(
                            "Density peak neighbor index is outside the candidate");
                    }
                    const std::size_t first = initial_root[i];
                    const std::size_t second = initial_root[j];
                    if (first == second) continue;
                    cross_basin_counts[row] = checked_add(
                        cross_basin_counts[row],
                        std::size_t{1},
                        "Density peak cross-basin edge count overflows size_t");
                    const core::Real lower_peak_density = std::min(
                        densities[first], densities[second]);
                    const core::Real saddle_density = std::min(
                        densities[i], densities[j]);
                    const core::Real merge_ratio =
                        saddle_density / lower_peak_density;
                    if (!std::isfinite(merge_ratio) || merge_ratio < 0.0) {
                        throw std::runtime_error(
                            "Density peak saddle ratio is invalid");
                    }
                    if (merge_ratio
                        >= options_.saddle_to_lower_peak_merge_ratio) {
                        qualifying_neighbors[row_begin + neighbor_rank] = j;
                    }
                }
            });

        for (std::size_t row = 0; row < batch_count; ++row) {
            observed_cross_basin_edges = checked_add(
                observed_cross_basin_edges,
                cross_basin_counts[row],
                "Density peak cross-basin edge count overflows size_t");
            const std::size_t i = batch_begin + row;
            const std::size_t row_begin = row * effective_k;
            for (std::size_t neighbor_rank = 0;
                 neighbor_rank < effective_k;
                 ++neighbor_rank) {
                const std::size_t j =
                    qualifying_neighbors[row_begin + neighbor_rank];
                if (j == invalid_index) continue;
                (void)unite_peak_roots(
                    merge_parent,
                    initial_root[i],
                    initial_root[j],
                    densities,
                    local);
            }
        }
    }
    if (observed_cross_basin_edges > plan.maximum_directed_knn_edges) {
        throw std::logic_error(
            "Density peak cross-basin edge count exceeds its execution plan");
    }

    std::vector<std::vector<std::size_t>> final_members(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t retained_peak = find_root(
            merge_parent, initial_root[i]);
        final_members[retained_peak].push_back(local[i].store_index);
    }
    for (auto& members : final_members) {
        std::sort(
            members.begin(), members.end(), [&](std::size_t lhs, std::size_t rhs) {
                return particle_ids[lhs] < particle_ids[rhs];
            });
    }

    DensityPeakDeblendResult result;
    result.candidate_id = candidate.id;
    result.effective_k_neighbors = effective_k;
    std::sort(
        peak_roots.begin(), peak_roots.end(), [&](std::size_t lhs, std::size_t rhs) {
            return local[lhs].id < local[rhs].id;
        });
    result.peaks.reserve(peak_roots.size());
    for (const std::size_t peak : peak_roots) {
        const std::size_t retained = find_root(merge_parent, peak);
        DensityPeakRecord record;
        record.candidate_id = candidate.id;
        record.peak_particle_id = local[peak].id;
        record.peak_particle_index = local[peak].store_index;
        record.position = local[peak].position;
        record.density = densities[peak];
        record.retained_as_host = retained == peak;
        if (retained != peak) {
            record.merged_into_peak_particle_id = local[retained].id;
        }
        result.peaks.push_back(std::move(record));
    }

    std::vector<std::size_t> retained_roots;
    retained_roots.reserve(peak_roots.size());
    for (std::size_t root = 0; root < final_members.size(); ++root) {
        if (!final_members[root].empty()) retained_roots.push_back(root);
    }
    std::sort(
        retained_roots.begin(), retained_roots.end(), [&](std::size_t lhs, std::size_t rhs) {
            return local[lhs].id < local[rhs].id;
        });
    result.hosts.reserve(retained_roots.size());
    for (std::size_t seed_id = 0; seed_id < retained_roots.size(); ++seed_id) {
        const std::size_t peak = retained_roots[seed_id];
        DeblendedHostSeed host;
        host.candidate_id = candidate.id;
        host.seed_id = seed_id;
        host.peak_particle_id = local[peak].id;
        host.peak_particle_index = local[peak].store_index;
        host.peak_position = local[peak].position;
        host.peak_density = densities[peak];
        host.particle_indices = std::move(final_members[peak]);
        host.meets_minimum_particle_count =
            host.particle_indices.size() >= options_.minimum_host_particles;
        result.hosts.push_back(std::move(host));
    }
    return result;
}

} // namespace cosmo_nbody::halo
