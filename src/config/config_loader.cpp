// Public vocabulary and override value kinds are owned by ConfigSchema.
#include "cosmo_nbody/config/config_loader.hpp"
#include "cosmo_nbody/config/config_schema.hpp"
#include "cosmo_nbody/io/content_hash.hpp"

#include <toml++/toml.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cosmo_nbody {
namespace config {

namespace {

constexpr std::size_t kMaximumConfigBytes = std::size_t{64} << 20;

std::string source_suffix(const toml::node& node) {
    std::ostringstream out;
    out << " at " << node.source().begin;
    return out.str();
}

[[noreturn]] void throw_config_error(
    const std::string& message,
    const toml::node& node) {
    throw std::runtime_error(message + source_suffix(node));
}

void reject_unknown_keys(const toml::table& tbl) {
    const auto& schema = ConfigSchema::instance();
    for (const auto& [top_key, top_node] : tbl) {
        const std::string section_name(top_key.str());
        if (!schema.has_section(section_name)) {
            throw_config_error(
                "Unknown or unsupported config section '" + section_name + "'",
                top_node);
        }
        const toml::table* section = top_node.as_table();
        if (!section) {
            throw_config_error(
                "Config section '" + section_name + "' must be a TOML table",
                top_node);
        }
        for (const auto& [key, node] : *section) {
            const std::string key_name(key.str());
            if (!schema.has_key(section_name, key_name)) {
                throw_config_error(
                    "Unknown or unsupported config key '"
                        + section_name + "." + key_name + "'",
                    node);
            }
        }
    }
}

const toml::node* get_optional(
    const toml::table* table,
    std::string_view key) {
    return table ? table->get(std::string(key)) : nullptr;
}

const toml::table& require_section(
    const toml::table& root,
    std::string_view section_name) {
    const toml::node* node = root.get(std::string(section_name));
    if (!node) {
        throw std::invalid_argument(
            "Scientific simulation configuration requires explicit ["
            + std::string(section_name) + "] section");
    }
    const toml::table* section = node->as_table();
    if (!section) {
        throw_config_error(
            "Config section '" + std::string(section_name)
                + "' must be a TOML table",
            *node);
    }
    return *section;
}

const toml::node& require_key(
    const toml::table& section,
    std::string_view section_name,
    std::string_view key) {
    const toml::node* node = section.get(std::string(key));
    if (!node) {
        throw std::invalid_argument(
            "Scientific simulation configuration requires explicit key '"
            + std::string(section_name) + "." + std::string(key) + "'");
    }
    return *node;
}

void require_keys(
    const toml::table& section,
    std::string_view section_name,
    std::initializer_list<std::string_view> keys) {
    for (const std::string_view key : keys) {
        (void)require_key(section, section_name, key);
    }
}

std::string require_string_key(
    const toml::table& section,
    std::string_view section_name,
    std::string_view key) {
    const toml::node& node = require_key(section, section_name, key);
    if (auto value = node.value<std::string>()) return *value;
    throw_config_error(
        "Config key '" + std::string(section_name) + "." + std::string(key)
            + "' must be a string",
        node);
}

void reject_key_if_present(
    const toml::table& section,
    std::string_view section_name,
    std::string_view key,
    std::string_view reason) {
    if (const toml::node* node = section.get(std::string(key))) {
        throw_config_error(
            "Config key '" + std::string(section_name) + "." + std::string(key)
                + "' " + std::string(reason),
            *node);
    }
}

void require_explicit_scientific_configuration(const toml::table& root) {
    const toml::table& cosmology = require_section(root, "cosmology");
    require_keys(
        cosmology,
        "cosmology",
        {"h", "omega_m", "omega_lambda", "omega_b", "sigma8", "n_s"});

    const toml::table& box = require_section(root, "box");
    require_keys(
        box,
        "box",
        {"comoving_size_Mpc_h", "particles_per_dimension", "pm_mesh_per_dimension"});

    const toml::table& gravity = require_section(root, "gravity");
    const std::string solver = require_string_key(gravity, "gravity", "solver");
    if (solver == "PM") {
        require_keys(gravity, "gravity", {"deconvolve_cic"});
    } else if (solver == "TreePM") {
        require_keys(
            gravity,
            "gravity",
            {"softening_comoving_Mpc_h", "theta", "split_scale_cells", "cutoff_multiplier"});
    }

    const toml::table& time = require_section(root, "time");
    require_keys(
        time,
        "time",
        {"start_redshift", "final_redshift", "delta_ln_a"});

    const toml::table& ic = require_section(root, "ic");
    const std::string ic_mode = require_string_key(ic, "ic", "mode");
    if (ic_mode == "generate") {
        require_keys(
            ic,
            "ic",
            {"lpt_order", "seed", "mesh_per_dimension", "power_spectrum_file",
             "power_spectrum_redshift", "power_spectrum_fidelity", "amplitude_mode",
             "phase_pairing"});
        reject_key_if_present(
            ic, "ic", "snapshot_file", "is valid only for ic.mode='snapshot'");
        reject_key_if_present(
            ic, "ic", "expected_snapshot_sha256", "is valid only for ic.mode='snapshot'");
    } else if (ic_mode == "snapshot") {
        require_keys(ic, "ic", {"snapshot_file"});
        for (const std::string_view key : {
                 "lpt_order", "seed", "mesh_per_dimension", "power_spectrum_file",
                 "power_spectrum_redshift", "power_spectrum_fidelity", "amplitude_mode",
                 "phase_pairing", "max_mode_per_axis"}) {
            reject_key_if_present(
                ic, "ic", key, "is a generated-IC parameter and must not be supplied for ic.mode='snapshot'");
        }
    }

    const toml::table& output = require_section(root, "output");
    (void)require_key(output, "output", "snapshot_scale_factors");
}

core::Real read_real(
    const toml::table* table,
    std::string_view section_name,
    std::string_view key,
    core::Real fallback) {
    const toml::node* node = get_optional(table, key);
    if (!node) return fallback;
    if (auto value = node->value<core::Real>()) return *value;
    if (auto integer = node->value<std::int64_t>()) {
        return static_cast<core::Real>(*integer);
    }
    throw_config_error(
        "Config key '" + std::string(section_name) + "." + std::string(key)
            + "' must be numeric",
        *node);
}

std::uint64_t read_uint64(
    const toml::table* table,
    std::string_view section_name,
    std::string_view key,
    std::uint64_t fallback) {
    const toml::node* node = get_optional(table, key);
    if (!node) return fallback;
    if (auto value = node->value<std::int64_t>()) {
        if (*value < 0) {
            throw_config_error(
                "Config key '" + std::string(section_name) + "."
                    + std::string(key) + "' must be non-negative",
                *node);
        }
        return static_cast<std::uint64_t>(*value);
    }
    throw_config_error(
        "Config key '" + std::string(section_name) + "." + std::string(key)
            + "' must be an integer",
        *node);
}

int read_int(
    const toml::table* table,
    std::string_view section_name,
    std::string_view key,
    int fallback) {
    const toml::node* node = get_optional(table, key);
    if (!node) return fallback;
    if (auto value = node->value<std::int64_t>()) {
        if (*value < static_cast<std::int64_t>(std::numeric_limits<int>::min())
            || *value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            throw_config_error(
                "Config key '" + std::string(section_name) + "."
                    + std::string(key) + "' does not fit the supported int range",
                *node);
        }
        return static_cast<int>(*value);
    }
    throw_config_error(
        "Config key '" + std::string(section_name) + "." + std::string(key)
            + "' must be an integer",
        *node);
}

bool read_bool(
    const toml::table* table,
    std::string_view section_name,
    std::string_view key,
    bool fallback) {
    const toml::node* node = get_optional(table, key);
    if (!node) return fallback;
    if (auto value = node->value<bool>()) return *value;
    throw_config_error(
        "Config key '" + std::string(section_name) + "." + std::string(key)
            + "' must be a boolean",
        *node);
}

std::string read_string(
    const toml::table* table,
    std::string_view section_name,
    std::string_view key,
    std::string fallback) {
    const toml::node* node = get_optional(table, key);
    if (!node) return fallback;
    if (auto value = node->value<std::string>()) {
        if (value->find('\0') != std::string::npos) {
            throw_config_error(
                "Config key '" + std::string(section_name) + "."
                    + std::string(key) + "' must be a NUL-free string",
                *node);
        }
        return *value;
    }
    throw_config_error(
        "Config key '" + std::string(section_name) + "." + std::string(key)
            + "' must be a string",
        *node);
}

void require_regular_input_file(
    const std::string& path,
    const char* config_key) {
    if (path.empty()) {
        throw std::invalid_argument(
            std::string(config_key) + " must not be empty");
    }
    std::error_code ec;
    const std::filesystem::path input(path);
    const bool regular = std::filesystem::is_regular_file(input, ec);
    if (ec || !regular) {
        throw std::invalid_argument(
            std::string(config_key)
            + " must name an existing regular file under the current runtime path semantics: '"
            + path + "'");
    }
}

void require_optional_sha256(
    const std::string& value,
    const char* config_key) {
    if (value.empty()) return;
    if (value.size() != 64U) {
        throw std::invalid_argument(
            std::string(config_key)
            + " must contain exactly 64 lowercase hexadecimal characters");
    }
    for (const char character : value) {
        const bool decimal = character >= '0' && character <= '9';
        const bool lower_hex = character >= 'a' && character <= 'f';
        if (!decimal && !lower_hex) {
            throw std::invalid_argument(
                std::string(config_key)
                + " must contain exactly 64 lowercase hexadecimal characters");
        }
    }
}

[[noreturn]] void throw_override_value_error(
    const std::string& dotted_key,
    const std::string& expected) {
    throw std::runtime_error(
        "Override '" + dotted_key + "' requires " + expected);
}

std::int64_t parse_signed_override(
    const std::string& dotted_key,
    const std::string& raw) {
    try {
        std::size_t consumed = 0;
        const long long value = std::stoll(raw, &consumed);
        if (consumed == raw.size()) return static_cast<std::int64_t>(value);
    } catch (...) {
    }
    throw_override_value_error(dotted_key, "a signed integer");
}

std::int64_t parse_unsigned_override(
    const std::string& dotted_key,
    const std::string& raw) {
    if (!raw.empty() && raw.front() == '-') {
        throw_override_value_error(dotted_key, "a non-negative integer");
    }
    try {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(raw, &consumed);
        if (consumed == raw.size()
            && value <= static_cast<unsigned long long>(
                std::numeric_limits<std::int64_t>::max())) {
            return static_cast<std::int64_t>(value);
        }
    } catch (...) {
    }
    throw_override_value_error(
        dotted_key,
        "a non-negative integer representable by TOML's signed integer domain");
}

core::Real parse_real_override(
    const std::string& dotted_key,
    const std::string& raw) {
    try {
        std::size_t consumed = 0;
        const core::Real value = std::stod(raw, &consumed);
        if (consumed == raw.size() && std::isfinite(value)) return value;
    } catch (...) {
    }
    throw_override_value_error(dotted_key, "a finite real number");
}

void insert_override_value(
    toml::table& table,
    const std::string& key,
    const std::string& dotted_key,
    const std::string& raw_value,
    ConfigValueKind kind) {
    switch (kind) {
        case ConfigValueKind::Boolean:
            if (raw_value == "true" || raw_value == "false") {
                table.insert_or_assign(key, raw_value == "true");
                return;
            }
            throw_override_value_error(dotted_key, "true or false");
        case ConfigValueKind::SignedInteger:
            table.insert_or_assign(
                key, parse_signed_override(dotted_key, raw_value));
            return;
        case ConfigValueKind::UnsignedInteger:
            table.insert_or_assign(
                key, parse_unsigned_override(dotted_key, raw_value));
            return;
        case ConfigValueKind::Real:
            table.insert_or_assign(
                key, parse_real_override(dotted_key, raw_value));
            return;
        case ConfigValueKind::String:
            if (raw_value.find('\0') != std::string::npos) {
                throw_override_value_error(dotted_key, "a NUL-free string");
            }
            table.insert_or_assign(key, raw_value);
            return;
        case ConfigValueKind::RealArray:
            throw std::runtime_error(
                "Override '" + dotted_key
                + "' is array-valued and cannot be changed by scalar --set");
    }
    throw std::logic_error("Unhandled ConfigValueKind");
}

void apply_overrides(
    toml::table& tbl,
    int argc,
    const char* const* argv) {
    if (!argv) return;
    const auto& schema = ConfigSchema::instance();
    for (int idx = 0; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        if (arg != "--set") continue;
        if (idx + 1 >= argc) {
            throw std::runtime_error(
                "Override flag --set requires section.key=value");
        }

        const std::string kv = argv[++idx];
        const auto pos = kv.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error(
                "Overrides must use --set section.key=value");
        }
        const std::string dotted_key = kv.substr(0, pos);
        const std::string raw_value = kv.substr(pos + 1);
        const auto dot_pos = dotted_key.find('.');
        if (dot_pos == std::string::npos
            || dot_pos == 0
            || dot_pos == dotted_key.size() - 1
            || dotted_key.find('.', dot_pos + 1) != std::string::npos) {
            throw std::runtime_error(
                "Overrides must use dotted keys: --set section.key=value");
        }

        const std::string group = dotted_key.substr(0, dot_pos);
        const std::string subkey = dotted_key.substr(dot_pos + 1);
        const ConfigKeySpec* spec = schema.find(group, subkey);
        if (!spec) {
            throw std::runtime_error(
                "Unknown override key '" + dotted_key + "'");
        }
        if (!spec->scalar_override_allowed()) {
            throw std::runtime_error(
                "Override '" + dotted_key
                + "' is not scalar and cannot be changed by --set");
        }

        toml::node* group_node = tbl.get(group);
        if (!group_node) {
            tbl.insert(group, toml::table{});
            group_node = tbl.get(group);
        }
        if (!group_node || !group_node->is_table()) {
            throw std::runtime_error(
                "Override section '" + group + "' must be a TOML table");
        }
        insert_override_value(
            *group_node->as_table(),
            subkey,
            dotted_key,
            raw_value,
            spec->kind);
    }
}

std::string read_config_bytes(const std::string& filepath) {
    std::ifstream input(filepath, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Failed to open config file: " + filepath);
    }

    std::string bytes;
    std::array<char, 64 * 1024> buffer{};
    for (;;) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            const std::size_t amount = static_cast<std::size_t>(count);
            if (amount > kMaximumConfigBytes - bytes.size()) {
                throw std::runtime_error(
                    "Configuration exceeds the 64 MiB admission bound");
            }
            bytes.append(buffer.data(), amount);
        }
        if (input.eof()) break;
        if (!input) {
            throw std::runtime_error(
                "I/O failure while reading config file: " + filepath);
        }
    }
    return bytes;
}

SimulationParameters build_parameters(
    toml::table tbl,
    int argc,
    const char* const* argv) {
    apply_overrides(tbl, argc, argv);
    reject_unknown_keys(tbl);
    require_explicit_scientific_configuration(tbl);

    CosmologyParams c;
    BoxParams b;
    GravityParams g;
    TimeParams t;
    ICParams i;
    OutputParams o;
    ValidationParams v;
    RuntimeParams r;
    MemoryPolicyParams m;

    if (auto cosmo = tbl["cosmology"].as_table()) {
        c.h = read_real(cosmo, "cosmology", "h", c.h);
        c.omega_m = read_real(cosmo, "cosmology", "omega_m", c.omega_m);
        c.omega_lambda = read_real(
            cosmo, "cosmology", "omega_lambda", c.omega_lambda);
        c.omega_b = read_real(cosmo, "cosmology", "omega_b", c.omega_b);
        c.sigma8 = read_real(cosmo, "cosmology", "sigma8", c.sigma8);
        c.n_s = read_real(cosmo, "cosmology", "n_s", c.n_s);
    }

    if (auto box = tbl["box"].as_table()) {
        b.L = read_real(box, "box", "comoving_size_Mpc_h", b.L);
        b.N = read_uint64(box, "box", "particles_per_dimension", b.N);
        b.N_mesh = read_uint64(
            box, "box", "pm_mesh_per_dimension", b.N_mesh);
    }

    if (auto gravity = tbl["gravity"].as_table()) {
        g.solver = read_string(gravity, "gravity", "solver", g.solver);
        if (g.solver == "PM") {
            for (const std::string_view key : {
                     "softening_comoving_Mpc_h",
                     "theta",
                     "split_scale_cells",
                     "cutoff_multiplier"}) {
                if (const toml::node* node = get_optional(gravity, key)) {
                    throw_config_error(
                        "Config key 'gravity." + std::string(key)
                            + "' is valid only for TreePM",
                        *node);
                }
            }
        }
        g.eps = read_real(
            gravity, "gravity", "softening_comoving_Mpc_h", g.eps);
        g.theta = read_real(gravity, "gravity", "theta", g.theta);
        g.split_scale_cells = read_real(
            gravity, "gravity", "split_scale_cells", g.split_scale_cells);
        g.cutoff_multiplier = read_real(
            gravity, "gravity", "cutoff_multiplier", g.cutoff_multiplier);
        if (g.solver == "TreePM") {
            if (const toml::node* node = get_optional(
                    gravity, "deconvolve_cic")) {
                const bool requested = read_bool(
                    gravity,
                    "gravity",
                    "deconvolve_cic",
                    true);
                if (!requested) {
                    throw_config_error(
                        "Config key 'gravity.deconvolve_cic' cannot be false for TreePM; the maintained TreePM PM leg compensates both CIC deposition and CIC force interpolation",
                        *node);
                }
            }
            g.deconvolve_cic = true;
        } else {
            g.deconvolve_cic = read_bool(
                gravity,
                "gravity",
                "deconvolve_cic",
                false);
        }
    }

    if (auto time = tbl["time"].as_table()) {
        t.z_start = read_real(time, "time", "start_redshift", t.z_start);
        t.z_final = read_real(time, "time", "final_redshift", t.z_final);
        t.delta_ln_a = read_real(time, "time", "delta_ln_a", t.delta_ln_a);
        t.step_policy = read_string(
            time, "time", "step_policy", t.step_policy);
    }

    if (auto ic = tbl["ic"].as_table()) {
        i.mode = read_string(ic, "ic", "mode", i.mode);
        if (i.mode == "generate") {
            // These keys are all required before this point. Literal fallbacks
            // are therefore unreachable; importantly, seed=0 and mesh=0 are
            // explicit user values rather than borrowed struct defaults.
            i.lpt_order = read_int(ic, "ic", "lpt_order", 0);
            i.seed = read_uint64(ic, "ic", "seed", 0);
            i.mesh_per_dimension = read_uint64(
                ic, "ic", "mesh_per_dimension", 0);
            if (ic->contains("max_mode_per_axis")) {
                i.max_mode_per_axis = read_uint64(
                    ic, "ic", "max_mode_per_axis", 0);
            }
            i.power_spectrum_file = read_string(
                ic, "ic", "power_spectrum_file", "");
            i.power_spectrum_redshift = read_real(
                ic,
                "ic",
                "power_spectrum_redshift",
                std::numeric_limits<core::Real>::quiet_NaN());
            i.power_spectrum_fidelity = read_string(
                ic, "ic", "power_spectrum_fidelity", "");
            i.amplitude_mode = read_string(
                ic, "ic", "amplitude_mode", "");
            i.phase_pairing = read_string(
                ic, "ic", "phase_pairing", "");
        } else if (i.mode == "snapshot") {
            i.snapshot_file = read_string(
                ic, "ic", "snapshot_file", "");
            i.expected_snapshot_sha256 = read_string(
                ic,
                "ic",
                "expected_snapshot_sha256",
                "");
        }
    }

    require_optional_sha256(
        i.expected_snapshot_sha256,
        "ic.expected_snapshot_sha256");
    if (!i.expected_snapshot_sha256.empty() && i.mode != "snapshot") {
        throw std::invalid_argument(
            "ic.expected_snapshot_sha256 is valid only when ic.mode = 'snapshot'");
    }

    // Interpret input paths exactly as runtime does: relative to process cwd.
    if (i.mode == "generate") {
        require_regular_input_file(
            i.power_spectrum_file, "ic.power_spectrum_file");
    } else if (i.mode == "snapshot") {
        // The serial reader or distributed rank-0 reader admits the exact file.
        // Config loading precedes MPI initialization; peers need not have the
        // snapshot in their local filesystem. Keep syntax checks here only.
        if (i.snapshot_file.empty()) {
            throw std::invalid_argument("ic.snapshot_file must not be empty");
        }
    }

    if (auto output = tbl["output"].as_table()) {
        if (const auto* s_node = get_optional(
                output, "snapshot_scale_factors")) {
            const auto* s_arr = s_node->as_array();
            if (!s_arr) {
                throw_config_error(
                    "Config key 'output.snapshot_scale_factors' must be an array",
                    *s_node);
            }
            o.snapshot_scale_factors.clear();
            for (const auto& elem : *s_arr) {
                if (auto value = elem.value<core::Real>()) {
                    o.snapshot_scale_factors.push_back(*value);
                } else if (auto integer = elem.value<std::int64_t>()) {
                    o.snapshot_scale_factors.push_back(
                        static_cast<core::Real>(*integer));
                } else {
                    throw_config_error(
                        "Config key 'output.snapshot_scale_factors' must contain only numbers",
                        elem);
                }
            }
        }

        if (get_optional(output, "format") != nullptr) {
            o.formats = {read_string(output, "output", "format", "hdf5")};
        }
        o.restart_cadence_steps = read_uint64(
            output,
            "output",
            "restart_cadence_steps",
            o.restart_cadence_steps);
        o.root_directory = read_string(
            output, "output", "root_directory", o.root_directory);
        o.run_label = read_string(
            output, "output", "run_label", o.run_label);
        o.timestamped_run_directory = read_bool(
            output,
            "output",
            "timestamped_run_directory",
            o.timestamped_run_directory);
        o.snapshot_batch_particles = read_uint64(
            output,
            "output",
            "snapshot_batch_particles",
            o.snapshot_batch_particles);
    }

    if (auto diagnostics = tbl["diagnostics"].as_table()) {
        v.write_diagnostics = read_bool(
            diagnostics,
            "diagnostics",
            "write_diagnostics",
            v.write_diagnostics);
    }

    if (auto runtime = tbl["runtime"].as_table()) {
        r.num_threads = read_uint64(
            runtime, "runtime", "num_threads", r.num_threads);
        r.mpi_enabled = read_bool(
            runtime, "runtime", "mpi_enabled", r.mpi_enabled);
    }

    if (auto memory = tbl["memory"].as_table()) {
        m.ic_scratch_mode = parse_scratch_mode(read_string(
            memory,
            "memory",
            "ic_scratch_mode",
            std::string(scratch_mode_name(m.ic_scratch_mode))));
        m.evolution_scratch_mode = parse_scratch_mode(read_string(
            memory,
            "memory",
            "evolution_scratch_mode",
            std::string(scratch_mode_name(m.evolution_scratch_mode))));
        m.scratch_directory = read_string(
            memory, "memory", "scratch_directory", m.scratch_directory);
    }

    return SimulationParameters(c, b, g, t, i, o, r, v, m);
}

} // namespace

LoadedSimulationConfig ConfigLoader::load_with_identity(
    const std::string& filepath,
    int argc,
    const char* const* argv) {
    const std::string bytes = read_config_bytes(filepath);

    toml::table tbl;
    try {
        tbl = toml::parse(bytes, filepath);
    } catch (const toml::parse_error& err) {
        std::cerr << "TOML parse error: " << err.description()
                  << " at " << err.source().begin << "\n";
        throw std::runtime_error("Failed to parse config file");
    }

    LoadedSimulationConfig loaded{
        build_parameters(std::move(tbl), argc, argv),
        io::sha256_text(bytes)};
    return loaded;
}

SimulationParameters ConfigLoader::load(
    const std::string& filepath,
    int argc,
    const char* const* argv) {
    return load_with_identity(filepath, argc, argv).parameters;
}

} // namespace config
} // namespace cosmo_nbody
