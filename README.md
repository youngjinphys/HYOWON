# HYOWON v0.0.1 Beta

C++20 cosmological N-body: PM/TreePM, 1LPT/2LPT, HDF5, optional MPI.
Flat matter–Λ background; one collisionless matter population. No hydrodynamics,
radiation or massive-neutrino evolution. Experimental. Accuracy and convergence
must be established for each application. File formats are not stable.

## Build

Linux/macOS. CMake ≥3.20, C++20 compiler, FFTW3, HDF5. Default: OpenMP and threaded
FFTW. CMake fetches pinned toml++ sources; provide network access or a local source.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
cmake --install build --prefix "$HOME/.local"
```

| CMake option | Default | Meaning |
|---|---|---|
| `HYOWON_ENABLE_OPENMP` | `ON` | CPU threading; requires an OpenMP runtime. |
| `HYOWON_ENABLE_FFTW_THREADS` | OpenMP setting | Threaded FFTW; requires OpenMP and matching FFTW threads library. |
| `HYOWON_ENABLE_MPI` | `OFF` | Distributed runtime; requires MPI C++. |
| `HYOWON_ENABLE_FFTW_MPI` | `OFF` | Distributed FFT; requires MPI enabled and matching FFTW-MPI. |
| `HYOWON_BUILD_ANALYZER` | `ON` | Build `hyowon_analyze`. |
| `HYOWON_OPENMP_ROOT` | empty | Optional OpenMP installation prefix, principally for Apple Clang. |
| `HYOWON_FFTW_ROOT` | empty | FFTW prefix; base, threads and MPI libraries must share a provider. |
| `FETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS` | unset | Local toml++ source in place of download. |
| `CMAKE_BUILD_TYPE` | `Release` | Build configuration for single-configuration generators. |
| `CMAKE_INSTALL_PREFIX` | CMake default | Installation prefix. |

Serial build: add `-DHYOWON_ENABLE_OPENMP=OFF -DHYOWON_ENABLE_FFTW_THREADS=OFF`.
MPI build: add `-DHYOWON_ENABLE_MPI=ON -DHYOWON_ENABLE_FFTW_MPI=ON`.
Use separate build directories for these configurations. Python is not required.

## Run

Commands use paths relative to the current working directory. Supply a real
linear spectrum as described below. Match cosmology, box, particle count and
starting redshift between generated ICs and the evolution configuration.

```sh
mkdir -m 700 run
cp examples/minimal_generate.toml run/generate.toml
cp examples/minimal_snapshot.toml run/snapshot.toml
# Set ic.power_spectrum_file in run/generate.toml.
build/hyowon_make_ic run/generate.toml --output run/ic.hdf5 --evidence run/ic.json
build/hyowon run/snapshot.toml --set ic.snapshot_file=run/ic.hdf5 \
  --set output.root_directory=run/output
```

`run/output/minimal_snapshot/` contains `snapshots/`, `checkpoints/` and
`diagnostics/`. Use a new `output.run_label` for each execution, including restart.

```sh
build/hyowon run/snapshot.toml --restart CHECKPOINT_DIRECTORY \
  --set ic.snapshot_file=run/ic.hdf5 --set output.run_label=resumed
```

MPI evolution uses the same configuration and overrides on every rank:

```sh
mpiexec -n 2 build/hyowon run/snapshot.toml \
  --set ic.snapshot_file=run/ic.hdf5 --set runtime.mpi_enabled=true
```

| Program | Argument | Meaning |
|---|---|---|
| All | `--help`, `-h` | Usage; invoke alone. |
| All | `--version` | Version and configure-time source identity; invoke alone. |
| `hyowon` | `CONFIG.toml` | Required configuration. |
| `hyowon` | `--restart DIR` | Checkpoint directory; matching physics configuration required. |
| `hyowon` | `--set section.key=value` | Repeatable scalar override; arrays must be edited in TOML. |
| `hyowon_make_ic` | `CONFIG.toml` | Required generated-IC configuration; no `--set` support. |
| `hyowon_make_ic` | `--output PATH` | Required native IC HDF5 destination. |
| `hyowon_make_ic` | `--evidence PATH` | Optional realized-IC JSON evidence; distinct from output. |
| `hyowon_make_ic` | `--overwrite` | Replace destinations; snapshot and evidence are published separately. |

## Configuration

Unknown sections/keys are errors. All physical values must be finite. `Required`
means no implicit scientific default. Lengths are comoving Mpc/h unless stated.

| TOML key | Default / domain | Meaning |
|---|---|---|
| `cosmology.h` | Required, >0 | Hubble parameter divided by 100 km/s/Mpc. |
| `cosmology.omega_m` | Required, >0 | Total matter density fraction. |
| `cosmology.omega_lambda` | Required, ≥0 | Vacuum fraction; `omega_m + omega_lambda = 1`. |
| `cosmology.omega_b` | Required, [0, omega_m] | Baryon fraction of critical density. |
| `cosmology.sigma8` | Required, ≥0 | Present-day linear normalization. |
| `cosmology.n_s` | Required | Primordial spectral index. |
| `box.comoving_size_Mpc_h` | Required, >0 | Periodic box side L. |
| `box.particles_per_dimension` | Required, integer >0 | N; total particle count N³. |
| `box.pm_mesh_per_dimension` | Required, integer >0 | PM mesh side. |
| `gravity.solver` | Required: `PM`, `TreePM` | Force solver. |
| `gravity.deconvolve_cic` | Required for PM: boolean | CIC deconvolution; forced true for TreePM. |
| `gravity.softening_comoving_Mpc_h` | Required for TreePM, >0 | Softening length. |
| `gravity.theta` | Required for TreePM, >0 | Tree opening parameter. |
| `gravity.split_scale_cells` | Required for TreePM, >0 | Split radius in PM cells. |
| `gravity.cutoff_multiplier` | Required for TreePM, >0 | Cutoff/split ratio; resulting cutoff must be <L/2. |
| `ic.mode` | Required: `generate`, `snapshot` | Initial-condition source. |
| `ic.lpt_order` | Required for generate: 1, 2 | LPT order. |
| `ic.seed` | Required for generate: uint64 | Seed; zero is valid. |
| `ic.mesh_per_dimension` | Required for generate: integer ≥0 | Zero selects N for 1LPT, 2N for 2LPT. Otherwise a multiple of N; 2LPT requires ≥2N. |
| `ic.max_mode_per_axis` | Omitted: floor((N−1)/2) | Optional Cartesian bound K>0, 2K<N; projects first-order density and final 2LPT source. |
| `ic.power_spectrum_file` | Required for generate | External spectrum path. |
| `ic.power_spectrum_redshift` | Required for generate, >−1 | Spectrum epoch, distinct from start redshift. |
| `ic.power_spectrum_fidelity` | Required for generate: `precision_boltzmann` | Input provenance declaration. |
| `ic.amplitude_mode` | Required for generate: `gaussian`, `fixed` | Fourier amplitude sampling. |
| `ic.phase_pairing` | Required for generate: `independent`, `pair_a`, `pair_b` | Phase pairing; hold other IC inputs fixed for a pair. |
| `ic.snapshot_file` | Required for snapshot | Native HDF5 particle input. |
| `ic.expected_snapshot_sha256` | Empty | Optional expected SHA-256 of the snapshot. |
| `time.start_redshift` | Required, >−1 | Start; must exceed final redshift. |
| `time.final_redshift` | Required, ≥0 | End. |
| `time.delta_ln_a` | Required, >0 | Global logarithmic scale-factor step. |
| `time.step_policy` | `global` | Only implemented policy. |
| `output.snapshot_scale_factors` | Required array; may be empty | Strictly increasing targets, a_start < a ≤ a_final. |
| `output.format` | `hdf5` | Only implemented format. |
| `output.restart_cadence_steps` | 0 | Checkpoint interval; zero disables periodic checkpoints. |
| `output.root_directory` | `runs` | Output parent. |
| `output.run_label` | Empty | Execution label. |
| `output.timestamped_run_directory` | `true` | Prefix execution directory with a UTC timestamp. |
| `output.snapshot_batch_particles` | 1048576, integer >0 | HDF5 write batch size. |
| `diagnostics.write_diagnostics` | `false` | Write runtime/conservation diagnostics. |
| `runtime.num_threads` | 0 | Automatic host-thread selection; positive values explicitly select count. |
| `runtime.mpi_enabled` | `false` | Enable distributed execution in an MPI build. |
| `memory.ic_scratch_mode` | `memory` | `memory` or `disk` for IC work arrays. |
| `memory.evolution_scratch_mode` | `memory` | `memory` or `disk` for supported evolution work arrays. |
| `memory.scratch_directory` | Empty | Disk scratch location; empty uses the system temporary directory. |

Omit generated-IC keys in snapshot mode. Omit snapshot keys in generate mode.
Disk scratch changes storage placement, not the numerical method or total memory guarantee.

## Linear spectrum

Two whitespace-separated columns: k [h/Mpc], P(k) [(Mpc/h)³]. Both positive and
finite; k strictly increasing. Required metadata comments for the example cosmology:

```text
# k_unit=h/Mpc
# power_unit=(Mpc/h)^3
# redshift=0
# species=cold_plus_baryon
# gauge=synchronous
# fidelity=precision_boltzmann
# normalization=sigma8_z0
# normalization_value=0.8
# nonlinear=false
# massive_neutrino_sum_eV=0
# h=0.7
# omega_m=0.3
# omega_b=0.05
# omega_lambda=0.7
# n_s=1
```

Append actual external-solver data. Metadata must describe that data. Amplitudes
are not automatically corrected to match sigma8. Cover at least
`2π/L ≤ k ≤ sqrt(3)*K*2π/L`, and a wider range sufficient for the sigma8 integral.
The examples contain no spectrum. Common-support comparisons must hold IC mesh,
K, seed, spectrum, cosmology, epochs, amplitude and pairing fixed.

## Analysis

The output directory must be new, under an existing parent writable only by its
owner. Example field-only analysis:

```sh
build/hyowon_analyze run/output/minimal_snapshot/snapshots/snapshot_1.hdf5 \
  --output-directory run/analysis --field-mesh 8 --bins 4 \
  --field-interlacing interlaced --field-shot-noise raw \
  --field-max-k-fraction-nyquist 0.5 --skip-halos
```

Estimator coordinates have no implicit scientific defaults. Optional products
are disabled unless requested. Integers below must be positive unless stated.

| Option | Values / requirement | Meaning |
|---|---|---|
| `SNAPSHOT.hdf5` | Required positional path | Native snapshot; candidate B in cross comparisons. |
| `--output-directory DIR` | Required | New execution directory. |
| `--analysis-threads N` | Optional; automatic if omitted | Host thread count. |
| `--analysis-memory-budget-gib G` | Optional, >0 | Selected bounded-workspace ceiling; not a global RSS or FFT cap. |
| `--field-mesh N` | Required | Density mesh side. |
| `--bins N` | Required | Field power-spectrum bins. |
| `--field-interlacing MODE` | Required: `interlaced`, `single-grid` | Density estimator. |
| `--field-shot-noise MODE` | Required: `raw`, `subtract-cic-aliased-poisson` | Subtraction requires `single-grid`. |
| `--field-max-k-fraction-nyquist F` | Required, 0<F≤1 | Field support; not an accuracy threshold. |
| `--skip-halos` | Flag | Disable halo products. Otherwise both FoF options are required. |
| `--fof-linking-length-b B` | >0 | Linking length / mean particle separation. |
| `--fof-min-particles N` | Integer | Minimum FoF membership. |
| `--standard-so` | Flag; requires FoF minimum ≥2 | Deblended spherical-overdensity products; requires all peak and shape settings. |
| `--peak-density-k-neighbors N` | With standard SO | Density-neighbor count. |
| `--deblended-min-particles N` | With standard SO | Minimum deblended membership. |
| `--peak-saddle-merge-ratio X` | With standard SO, 0≤X≤1 | Peak/saddle merge criterion. |
| `--hmf-min-mass M` | >0; requires halos | HMF lower mass [Msun/h]. |
| `--hmf-max-mass M` | >minimum | HMF upper mass [Msun/h]. |
| `--hmf-bins N` | Required with both HMF limits | HMF bins. |
| `--xi-min-radius R` | ≥0; requires halos | Correlation lower radius [Mpc/h]. |
| `--xi-max-radius R` | >minimum | Correlation upper radius [Mpc/h]. |
| `--xi-bins N` | Required with both radii | Correlation bins. |
| `--xi-binning MODE` | `linear`, `logarithmic` | Required with radii; logarithmic requires minimum >0. |
| `--bispectrum` | Flag | Equilateral shell bispectrum. |
| `--bispectrum-bins N` | Required with bispectrum | Shell count. |
| `--bispectrum-max-k-fraction-nyquist F` | Required with bispectrum, 0<F<2/3 | Prevents FFT triangle-closure wraparound. |
| `--progenitor-snapshot PATH` | Requires halos | Earlier snapshot for links. |
| `--write-shapes` | Flag; requires halos | Shape products. |
| `--shape-tensor MODE` | `standard`, `reduced` | Required with shapes or standard SO. |
| `--shape-max-iterations N` | Required with shapes or standard SO | Iteration bound. |
| `--shape-convergence-tolerance X` | >0; required with shapes or standard SO | Shape convergence tolerance. |
| `--tidal-web` | Flag | Tidal-web classification. |
| `--tidal-web-smoothing-radius-mpc-h R` | Required with tidal web, ≥0 | Gaussian radius; zero means unsmoothed. |
| `--tidal-web-lambda-threshold X` | Required with tidal web; finite | Eigenvalue threshold. |
| `--cross-snapshot PATH` | Optional | Reference A; requires cross mesh. |
| `--cross-mesh N` | Required with cross snapshot | Independent comparison mesh. |
| `--cross-phase-space-summary` | Requires cross snapshot | Particle phase-space summary. |

## Environment and distribution

| Variable / command | Meaning |
|---|---|
| `OMP_NUM_THREADS`, `OMP_THREAD_LIMIT` | OpenMP runtime limits used by automatic thread selection. |
| `HYOWON_FFTW_WISDOM_DIRECTORY` | Optional existing absolute, non-symlink directory for FFTW wisdom. Unset disables explicit persistence; empty is invalid. |
| `cmake --build build --target package_source` | Source `.tar.gz` and `.zip` archives in the build directory. |

Reconfigure after source changes. `--version` describes configure-time source,
not a binary attestation; source archives without Git metadata report `unbound`.

[GPL-3.0-or-later](LICENSE). [Third-party notices](THIRD_PARTY_NOTICES.md).
