#include "cosmo_nbody/analysis/two_point_correlation.hpp"

#include "cosmo_nbody/analysis/spectral_numeric.hpp"
#include "cosmo_nbody/math/periodic_box.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef COSMO_NBODY_HAS_OPENMP
#include <omp.h>
#endif

namespace cosmo_nbody::analysis {
namespace {

constexpr std::size_t invalid_bin = std::numeric_limits<std::size_t>::max();
constexpr std::size_t maximum_cells_per_dimension = 1024;

struct BinLayout {
    std::vector<core::Real> edges;
    std::vector<core::Real> midpoints;

    std::size_t locate(core::Real radius) const noexcept {
        if (!std::isfinite(radius)
            || radius < edges.front()
            || radius >= edges.back()) {
            return invalid_bin;
        }
        const auto upper = std::upper_bound(edges.begin(), edges.end(), radius);
        if (upper == edges.begin() || upper == edges.end()) return invalid_bin;
        return static_cast<std::size_t>(std::distance(edges.begin(), upper) - 1);
    }
};

struct PairCounts {
    std::vector<std::size_t> bins;
    std::size_t in_range{0};
};

struct CellGrid {
    std::size_t cells_per_dimension{1};
    core::Real cell_width{0.0};
    int neighbor_reach{1};
    std::unordered_map<std::size_t, std::vector<std::size_t>> occupied;
    std::unordered_map<std::size_t, std::vector<std::size_t>> neighbor_cells;
    std::vector<std::size_t> point_cell_ids;
};

void validate_options(core::Real box_size, const TwoPointOptions& options) {
    if (!std::isfinite(box_size) || box_size <= 0.0) {
        throw std::invalid_argument(
            "TwoPointCorrelation box_size must be finite and positive");
    }
    if (!std::isfinite(options.min_radius)
        || !std::isfinite(options.max_radius)
        || options.min_radius < 0.0
        || options.max_radius <= options.min_radius
        || options.num_bins <= 0) {
        throw std::invalid_argument(
            "TwoPointCorrelation requires 0 <= min_radius < max_radius and positive num_bins");
    }
    if (options.binning == TwoPointBinning::Logarithmic
        && options.min_radius <= 0.0) {
        throw std::invalid_argument(
            "TwoPointCorrelation logarithmic binning requires min_radius > 0");
    }
    const core::Real half_box = 0.5 * box_size;
    const core::Real maximum = core::scale_safe_norm3(
        half_box, half_box, half_box);
    if (options.max_radius > maximum) {
        throw std::invalid_argument(
            "TwoPointCorrelation max_radius exceeds sqrt(3)*box_size/2");
    }
}

void validate_periodic_analytic_options(
    core::Real box_size,
    const TwoPointOptions& options) {
    validate_options(box_size, options);
    if (options.max_radius > 0.5 * box_size) {
        throw std::invalid_argument(
            "Periodic analytic TwoPointCorrelation requires max_radius <= box_size/2; larger radii need the exact piecewise cubic-torus shell measure");
    }
}

std::vector<core::Vec3> wrapped_points(
    std::span<const core::Vec3> points,
    core::Real box_size,
    const char* role) {
    std::vector<core::Vec3> output;
    output.reserve(points.size());
    for (const auto& point : points) {
        if (!std::isfinite(point.x)
            || !std::isfinite(point.y)
            || !std::isfinite(point.z)) {
            throw std::invalid_argument(
                std::string("TwoPointCorrelation ") + role
                + " coordinates must be finite");
        }
        output.push_back(math::wrap(point, box_size));
    }
    return output;
}

BinLayout make_layout(const TwoPointOptions& options) {
    BinLayout layout;
    const auto bin_count = static_cast<std::size_t>(options.num_bins);
    layout.edges.resize(bin_count + 1);
    layout.midpoints.resize(bin_count);
    if (options.binning == TwoPointBinning::Linear) {
        const core::Real width =
            (options.max_radius - options.min_radius)
            / static_cast<core::Real>(options.num_bins);
        if (!std::isfinite(width) || width <= 0.0) {
            throw std::overflow_error(
                "TwoPointCorrelation linear bin width is invalid");
        }
        for (std::size_t index = 0; index <= bin_count; ++index) {
            layout.edges[index] = index == bin_count
                ? options.max_radius
                : options.min_radius + static_cast<core::Real>(index) * width;
        }
        for (std::size_t index = 0; index < bin_count; ++index) {
            layout.midpoints[index] = std::midpoint(
                layout.edges[index], layout.edges[index + 1]);
        }
    } else {
        const core::Real log_span = detail::stable_positive_log_ratio(
            options.max_radius, options.min_radius);
        const core::Real width =
            log_span / static_cast<core::Real>(options.num_bins);
        if (!std::isfinite(width) || width <= 0.0) {
            throw std::overflow_error(
                "TwoPointCorrelation logarithmic bin width is invalid");
        }
        for (std::size_t index = 0; index <= bin_count; ++index) {
            const core::Real fraction = static_cast<core::Real>(index)
                / static_cast<core::Real>(options.num_bins);
            layout.edges[index] = detail::stable_positive_log_interpolate(
                options.min_radius, options.max_radius, fraction);
        }
        for (std::size_t index = 0; index < bin_count; ++index) {
            layout.midpoints[index] = std::sqrt(layout.edges[index])
                * std::sqrt(layout.edges[index + 1]);
        }
    }

    for (std::size_t index = 0; index < bin_count; ++index) {
        if (!std::isfinite(layout.edges[index])
            || !std::isfinite(layout.edges[index + 1])
            || !(layout.edges[index + 1] > layout.edges[index])
            || !std::isfinite(layout.midpoints[index])
            || layout.midpoints[index] < layout.edges[index]
            || layout.midpoints[index] > layout.edges[index + 1]) {
            throw std::overflow_error(
                "TwoPointCorrelation bin layout is not representable");
        }
    }
    return layout;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* label) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(std::string(label) + " pair-count overflow");
    }
    return lhs + rhs;
}

std::size_t checked_cell_id(
    std::size_t x,
    std::size_t y,
    std::size_t z,
    std::size_t n) {
    if (x >= n || y >= n || z >= n) {
        throw std::logic_error("TwoPointCorrelation cell coordinate is out of range");
    }
    if (x > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error("TwoPointCorrelation cell-id overflow");
    }
    const std::size_t xy = x * n + y;
    if (xy > std::numeric_limits<std::size_t>::max() / n) {
        throw std::overflow_error("TwoPointCorrelation cell-id overflow");
    }
    return xy * n + z;
}

void decode_cell_id(
    std::size_t id,
    std::size_t n,
    std::size_t& x,
    std::size_t& y,
    std::size_t& z) noexcept {
    z = id % n;
    id /= n;
    y = id % n;
    x = id / n;
}

std::size_t periodic_cell_index(long long value, std::size_t n) {
    const auto signed_n = static_cast<long long>(n);
    long long wrapped = value % signed_n;
    if (wrapped < 0) wrapped += signed_n;
    return static_cast<std::size_t>(wrapped);
}

std::vector<std::size_t> neighbor_cell_ids(
    std::size_t id,
    const CellGrid& grid) {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t z = 0;
    decode_cell_id(id, grid.cells_per_dimension, x, y, z);

    const int reach = grid.neighbor_reach;
    const std::size_t side = static_cast<std::size_t>(2 * reach + 1);
    if (side > std::numeric_limits<std::size_t>::max() / side
        || side * side > std::numeric_limits<std::size_t>::max() / side) {
        throw std::overflow_error(
            "TwoPointCorrelation neighbor-cell count overflow");
    }

    std::vector<std::size_t> ids;
    ids.reserve(side * side * side);
    for (int dx = -reach; dx <= reach; ++dx) {
        for (int dy = -reach; dy <= reach; ++dy) {
            for (int dz = -reach; dz <= reach; ++dz) {
                ids.push_back(checked_cell_id(
                    periodic_cell_index(
                        static_cast<long long>(x) + dx,
                        grid.cells_per_dimension),
                    periodic_cell_index(
                        static_cast<long long>(y) + dy,
                        grid.cells_per_dimension),
                    periodic_cell_index(
                        static_cast<long long>(z) + dz,
                        grid.cells_per_dimension),
                    grid.cells_per_dimension));
            }
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

CellGrid build_cell_grid(
    std::span<const core::Vec3> points,
    core::Real box_size,
    core::Real max_radius) {
    CellGrid grid;
    const core::Real desired_ratio = box_size / max_radius;
    std::size_t desired = 1;
    if (!std::isfinite(desired_ratio)
        || desired_ratio >= static_cast<core::Real>(maximum_cells_per_dimension)) {
        desired = maximum_cells_per_dimension;
    } else if (desired_ratio >= 1.0) {
        desired = static_cast<std::size_t>(std::floor(desired_ratio));
    }
    grid.cells_per_dimension = std::max<std::size_t>(1, desired);
    grid.cell_width = box_size
        / static_cast<core::Real>(grid.cells_per_dimension);
    if (!std::isfinite(grid.cell_width) || grid.cell_width <= 0.0) {
        throw std::overflow_error(
            "TwoPointCorrelation cell width is invalid");
    }
    const core::Real reach_real = std::ceil(max_radius / grid.cell_width);
    if (!std::isfinite(reach_real)
        || reach_real < 1.0
        || reach_real > static_cast<core::Real>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "TwoPointCorrelation neighbor reach is invalid");
    }
    grid.neighbor_reach = static_cast<int>(reach_real);
    grid.occupied.reserve(points.size());
    grid.point_cell_ids.resize(points.size());

    const auto coordinate_to_index = [&](core::Real coordinate) {
        const core::Real raw = std::floor(coordinate / grid.cell_width);
        if (!std::isfinite(raw) || raw < 0.0) {
            throw std::runtime_error(
                "TwoPointCorrelation wrapped coordinate produced an invalid cell");
        }
        const auto index = static_cast<std::size_t>(raw);
        return std::min(index, grid.cells_per_dimension - 1);
    };

    for (std::size_t index = 0; index < points.size(); ++index) {
        const std::size_t id = checked_cell_id(
            coordinate_to_index(points[index].x),
            coordinate_to_index(points[index].y),
            coordinate_to_index(points[index].z),
            grid.cells_per_dimension);
        grid.point_cell_ids[index] = id;
        grid.occupied[id].push_back(index);
    }

    grid.neighbor_cells.reserve(grid.occupied.size());
    for (const auto& [id, ignored] : grid.occupied) {
        (void)ignored;
        grid.neighbor_cells.emplace(id, neighbor_cell_ids(id, grid));
    }
    return grid;
}

std::size_t resolved_workers(std::size_t work_items) {
    if (work_items == 0) return 1;
#ifdef COSMO_NBODY_HAS_OPENMP
    const int requested = std::max(1, omp_get_max_threads());
    return std::min(work_items, static_cast<std::size_t>(requested));
#else
    return 1;
#endif
}

template <typename PairVisitor>
PairCounts parallel_count(
    std::size_t outer_count,
    std::size_t bin_count,
    PairVisitor&& visit,
    const char* label) {
    const std::size_t workers = resolved_workers(outer_count);
    std::vector<std::vector<std::size_t>> local_bins(
        workers, std::vector<std::size_t>(bin_count, 0));
    std::vector<std::size_t> local_totals(workers, 0);
    int failed = 0;
#ifdef COSMO_NBODY_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(workers)) reduction(|:failed)
#endif
    {
#ifdef COSMO_NBODY_HAS_OPENMP
        const std::size_t worker = static_cast<std::size_t>(omp_get_thread_num());
#else
        const std::size_t worker = 0;
#endif
#ifdef COSMO_NBODY_HAS_OPENMP
#pragma omp for schedule(static)
#endif
        for (std::size_t outer = 0; outer < outer_count; ++outer) {
            if (!visit(outer, local_bins[worker], local_totals[worker])) {
                failed = 1;
            }
        }
    }
    if (failed != 0) {
        throw std::overflow_error(std::string(label) + " pair-count overflow");
    }

    PairCounts result{std::vector<std::size_t>(bin_count, 0), 0};
    for (std::size_t worker = 0; worker < workers; ++worker) {
        result.in_range = checked_add(result.in_range, local_totals[worker], label);
        for (std::size_t bin = 0; bin < bin_count; ++bin) {
            result.bins[bin] = checked_add(
                result.bins[bin], local_bins[worker][bin], label);
        }
    }
    return result;
}

bool record_distance(
    const core::Vec3& first,
    const core::Vec3& second,
    core::Real box_size,
    const BinLayout& layout,
    std::vector<std::size_t>& bins,
    std::size_t& total) {
    const core::Real dx = math::minimum_image(first.x - second.x, box_size);
    const core::Real dy = math::minimum_image(first.y - second.y, box_size);
    const core::Real dz = math::minimum_image(first.z - second.z, box_size);
    const core::Real distance = core::scale_safe_norm3(dx, dy, dz);
    const std::size_t bin = layout.locate(distance);
    if (bin == invalid_bin) return true;
    if (bins[bin] == std::numeric_limits<std::size_t>::max()
        || total == std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    ++bins[bin];
    ++total;
    return true;
}

PairCounts count_auto(
    std::span<const core::Vec3> points,
    core::Real box_size,
    const BinLayout& layout,
    const CellGrid& grid,
    const char* label) {
    return parallel_count(
        points.size(),
        layout.midpoints.size(),
        [&](std::size_t first,
            std::vector<std::size_t>& bins,
            std::size_t& total) {
            const std::size_t cell = grid.point_cell_ids[first];
            const auto neighbor_it = grid.neighbor_cells.find(cell);
            if (neighbor_it == grid.neighbor_cells.end()) return false;
            for (const std::size_t neighbor : neighbor_it->second) {
                const auto occupied_it = grid.occupied.find(neighbor);
                if (occupied_it == grid.occupied.end()) continue;
                for (const std::size_t second : occupied_it->second) {
                    if (second <= first) continue;
                    if (!record_distance(
                            points[first], points[second], box_size,
                            layout, bins, total)) {
                        return false;
                    }
                }
            }
            return true;
        },
        label);
}

PairCounts count_cross(
    std::span<const core::Vec3> data,
    std::span<const core::Vec3> random_points,
    core::Real box_size,
    const BinLayout& layout,
    const CellGrid& data_grid,
    const CellGrid& random_grid) {
    if (data_grid.cells_per_dimension != random_grid.cells_per_dimension
        || data_grid.neighbor_reach != random_grid.neighbor_reach) {
        throw std::logic_error(
            "TwoPointCorrelation data/random cell grids are inconsistent");
    }
    return parallel_count(
        data.size(),
        layout.midpoints.size(),
        [&](std::size_t first,
            std::vector<std::size_t>& bins,
            std::size_t& total) {
            const std::size_t cell = data_grid.point_cell_ids[first];
            const auto neighbor_it = data_grid.neighbor_cells.find(cell);
            if (neighbor_it == data_grid.neighbor_cells.end()) return false;
            for (const std::size_t neighbor : neighbor_it->second) {
                const auto occupied_it = random_grid.occupied.find(neighbor);
                if (occupied_it == random_grid.occupied.end()) continue;
                for (const std::size_t second : occupied_it->second) {
                    if (!record_distance(
                            data[first], random_points[second], box_size,
                            layout, bins, total)) {
                        return false;
                    }
                }
            }
            return true;
        },
        "DR");
}

core::Real normalized(std::size_t count, long double denominator) {
    if (!std::isfinite(denominator) || denominator <= 0.0L) {
        throw std::overflow_error("TwoPointCorrelation normalization is invalid");
    }
    const long double value = static_cast<long double>(count) / denominator;
    if (!std::isfinite(value)
        || value > static_cast<long double>(std::numeric_limits<core::Real>::max())) {
        throw std::overflow_error("TwoPointCorrelation normalized pair count is invalid");
    }
    return static_cast<core::Real>(value);
}

core::Real periodic_shell_probability(
    core::Real radius_low,
    core::Real radius_high,
    core::Real box_size) {
    const long double lo = static_cast<long double>(radius_low)
        / static_cast<long double>(box_size);
    const long double hi = static_cast<long double>(radius_high)
        / static_cast<long double>(box_size);
    const long double shell = (hi - lo)
        * (hi * hi + hi * lo + lo * lo);
    const long double probability =
        (4.0L * std::numbers::pi_v<long double> / 3.0L) * shell;
    if (!std::isfinite(probability) || !(probability > 0.0L)
        || probability > 1.0L) {
        throw std::overflow_error(
            "TwoPointCorrelation periodic shell probability is invalid");
    }
    const core::Real narrowed = static_cast<core::Real>(probability);
    if (!std::isfinite(narrowed) || !(narrowed > 0.0)) {
        throw std::overflow_error(
            "TwoPointCorrelation periodic shell probability is not representable");
    }
    return narrowed;
}

} // namespace

TwoPointResult TwoPointCorrelation::compute(
    std::span<const core::Vec3> points,
    std::span<const core::Vec3> random_points,
    core::Real box_size,
    const TwoPointOptions& options) {
    validate_options(box_size, options);
    const auto data = wrapped_points(points, box_size, "data");
    const auto random = wrapped_points(random_points, box_size, "random");
    if (data.size() < 2) {
        throw std::invalid_argument(
            "TwoPointCorrelation data catalog requires at least two selected points");
    }
    if (random.size() < 2) {
        throw std::invalid_argument(
            "TwoPointCorrelation explicit random catalog requires at least two points");
    }

    const BinLayout layout = make_layout(options);
    const CellGrid data_grid = build_cell_grid(
        data, box_size, options.max_radius);
    const CellGrid random_grid = build_cell_grid(
        random, box_size, options.max_radius);
    const PairCounts dd = count_auto(
        data, box_size, layout, data_grid, "DD");
    const PairCounts dr = count_cross(
        data, random, box_size, layout, data_grid, random_grid);
    const PairCounts rr = count_auto(
        random, box_size, layout, random_grid, "RR");

    const long double nd = static_cast<long double>(data.size());
    const long double nr = static_cast<long double>(random.size());
    const long double dd_denominator = nd * (nd - 1.0L) / 2.0L;
    const long double dr_denominator = nd * nr;
    const long double rr_denominator = nr * (nr - 1.0L) / 2.0L;

    TwoPointResult result;
    result.box_size = box_size;
    result.input_point_count = points.size();
    result.used_point_count = data.size();
    result.random_point_count = random.size();
    result.data_data_pair_count_in_range = dd.in_range;
    result.data_random_pair_count_in_range = dr.in_range;
    result.random_random_pair_count_in_range = rr.in_range;
    result.pair_count_in_range = dd.in_range;
    result.random_catalog = "caller_supplied";
    result.tracer_label = options.tracer_label;
    result.binning = options.binning;
    result.estimator = "landy_szalay_explicit_random";
    result.reference_measure = "caller_supplied_selection_window";
    result.bins.resize(layout.midpoints.size());

    for (std::size_t index = 0; index < result.bins.size(); ++index) {
        auto& bin = result.bins[index];
        bin.radius_low = layout.edges[index];
        bin.radius_high = layout.edges[index + 1];
        bin.radius_midpoint = layout.midpoints[index];
        bin.data_data_pair_count = dd.bins[index];
        bin.data_random_pair_count = dr.bins[index];
        bin.random_random_pair_count = rr.bins[index];
        bin.data_data_normalized = normalized(dd.bins[index], dd_denominator);
        bin.data_random_normalized = normalized(dr.bins[index], dr_denominator);
        bin.random_random_normalized = normalized(rr.bins[index], rr_denominator);
        bin.xi = bin.random_random_normalized > 0.0
            ? (bin.data_data_normalized - 2.0 * bin.data_random_normalized
                + bin.random_random_normalized)
                / bin.random_random_normalized
            : std::numeric_limits<core::Real>::quiet_NaN();
    }
    return result;
}

TwoPointResult TwoPointCorrelation::compute(
    std::span<const core::Vec3> points,
    core::Real box_size,
    const TwoPointOptions& options) {
    validate_periodic_analytic_options(box_size, options);
    const auto data = wrapped_points(points, box_size, "data");
    if (data.size() < 2) {
        throw std::invalid_argument(
            "TwoPointCorrelation data catalog requires at least two selected points");
    }

    const BinLayout layout = make_layout(options);
    const CellGrid data_grid = build_cell_grid(
        data, box_size, options.max_radius);
    const PairCounts dd = count_auto(
        data, box_size, layout, data_grid, "DD");

    const long double nd = static_cast<long double>(data.size());
    const long double dd_denominator = nd * (nd - 1.0L) / 2.0L;

    TwoPointResult result;
    result.box_size = box_size;
    result.input_point_count = points.size();
    result.used_point_count = data.size();
    result.data_data_pair_count_in_range = dd.in_range;
    result.pair_count_in_range = dd.in_range;
    result.random_catalog = "none";
    result.tracer_label = options.tracer_label;
    result.binning = options.binning;
    result.estimator = "periodic_analytic_shell";
    result.reference_measure = "exact_uniform_periodic_shell_rmax_le_half_box";
    result.bins.resize(layout.midpoints.size());

    for (std::size_t index = 0; index < result.bins.size(); ++index) {
        auto& bin = result.bins[index];
        bin.radius_low = layout.edges[index];
        bin.radius_high = layout.edges[index + 1];
        bin.radius_midpoint = layout.midpoints[index];
        bin.data_data_pair_count = dd.bins[index];
        bin.data_data_normalized = normalized(dd.bins[index], dd_denominator);
        bin.expected_pair_probability = periodic_shell_probability(
            bin.radius_low, bin.radius_high, box_size);
        bin.xi = bin.data_data_normalized / bin.expected_pair_probability - 1.0;
        if (!std::isfinite(bin.xi)) {
            throw std::overflow_error(
                "TwoPointCorrelation periodic xi is not representable");
        }
    }
    return result;
}

} // namespace cosmo_nbody::analysis
