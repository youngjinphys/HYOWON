# HYOWON

HYOWON 0.0.3 Beta is a C++20 cosmological N-body research implementation for a
periodic, collisionless, one-fluid matter–Λ model. It provides pure PM, an
experimental TreePM path, 1LPT/2LPT initial conditions, native HDF5 snapshots and
restart, optional MPI/FFTW-MPI execution, and a standalone analysis program.

HYOWON does not model hydrodynamics, radiation evolution, baryonic feedback, or
massive-neutrino clustering. Numerical accuracy and convergence are properties of
a specified calculation, not of the version label alone.

## Requirements

- CMake 3.20 or newer
- C++20 compiler
- FFTW3
- HDF5 C library
- OpenMP by default
- MPI and FFTW-MPI only when distributed execution is enabled

`toml++` is fetched from a pinned upstream commit unless
`FETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS` points to a local source tree.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
cmake --install build --prefix "$HOME/.local"
```

Useful options:

| CMake option | Default | Meaning |
|---|---|---|
| `HYOWON_ENABLE_OPENMP` | `ON` | Enable OpenMP host threading. |
| `HYOWON_ENABLE_FFTW_THREADS` | OpenMP setting | Use threaded FFTW plans. |
| `HYOWON_ENABLE_MPI` | `OFF` | Enable MPI runtime support. |
| `HYOWON_ENABLE_FFTW_MPI` | `OFF` | Enable distributed FFTW; requires MPI. |
| `HYOWON_BUILD_ANALYZER` | `ON` | Build `hyowon_analyze`. |
| `HYOWON_OPENMP_ROOT` | empty | Optional OpenMP installation prefix. |
| `HYOWON_FFTW_ROOT` | empty | Optional coherent FFTW installation prefix. |

For a serial CPU build:

```sh
cmake -S . -B build-serial \
  -DHYOWON_ENABLE_OPENMP=OFF \
  -DHYOWON_ENABLE_FFTW_THREADS=OFF
```

For MPI + FFTW-MPI use a separate build directory and enable both MPI options.
HYOWON-owned C++ targets are compiled with floating-point reassociation disabled;
compensated reductions and exact-boundary predicates rely on ordinary IEEE-754
evaluation order.

## Quick start

The examples intentionally contain no linear power-spectrum data. Supply a real
external spectrum matching the configured cosmology.

```sh
mkdir -m 700 run
cp examples/minimal_generate.toml run/generate.toml
cp examples/minimal_snapshot.toml run/snapshot.toml
# Set ic.power_spectrum_file in run/generate.toml.

build/hyowon_make_ic run/generate.toml \
  --output run/ic.hdf5 --evidence run/ic.json

build/hyowon run/snapshot.toml \
  --set ic.snapshot_file=run/ic.hdf5 \
  --set output.root_directory=run/output
```

Keep the original IC file. Restart reopens it to verify the bound source identity.
Use a new `output.run_label` for each execution.

```sh
build/hyowon run/snapshot.toml --restart CHECKPOINT_DIRECTORY \
  --set ic.snapshot_file=run/ic.hdf5 \
  --set output.run_label=resumed
```

MPI uses the same configuration. A multi-rank launch requires an MPI build and
`runtime.mpi_enabled=true`; startup configuration and restart identity are agreed
collectively before rank-local simulation state is constructed.

```sh
mpiexec -n 2 build/hyowon run/snapshot.toml \
  --set ic.snapshot_file=run/ic.hdf5 \
  --set runtime.mpi_enabled=true
```

## Model and configuration

Unknown TOML sections and keys are errors. Physical values must be finite. Lengths
are comoving Mpc/h unless stated otherwise.

Core coordinates:

| TOML key | Requirement / meaning |
|---|---|
| `cosmology.h` | Hubble parameter divided by 100 km/s/Mpc, >0. |
| `cosmology.omega_m` | Total matter density fraction, >0. |
| `cosmology.omega_lambda` | Vacuum fraction, ≥0; flatness requires `omega_m + omega_lambda = 1`. |
| `cosmology.omega_b` | Baryon fraction of critical density, `0 <= omega_b <= omega_m`. |
| `cosmology.sigma8` | Present-day linear normalization, ≥0. |
| `cosmology.n_s` | Primordial spectral index. |
| `box.comoving_size_Mpc_h` | Periodic box side, >0. |
| `box.particles_per_dimension` | Particle lattice side N; total count N³. |
| `box.pm_mesh_per_dimension` | PM mesh side. |
| `gravity.solver` | `PM` or `TreePM`; TreePM remains experimental in this beta. |
| `gravity.deconvolve_cic` | Required for PM; TreePM forces it on. |
| `time.start_redshift` | Start epoch, greater than final redshift. |
| `time.final_redshift` | Final epoch, ≥0. |
| `time.delta_ln_a` | Global logarithmic scale-factor step, >0. |
| `output.snapshot_scale_factors` | Strictly increasing output targets in `(a_start, a_final]`. |
| `output.snapshot_batch_particles` | I/O batch upper bound. Distributed snapshot staging and verification additionally cap the effective batch at the largest actual rank payload; this does not alter numerical state. |
| `runtime.num_threads` | `0` selects automatically; positive values select explicitly. |
| `runtime.mpi_enabled` | Enables distributed execution in an MPI build. |
| `memory.ic_scratch_mode` | `memory` or `disk`. |
| `memory.evolution_scratch_mode` | `memory` or `disk` for supported evolution workspaces. |
| `memory.scratch_directory` | Optional disk-scratch directory. |

Disk scratch changes backing placement. It is not a guarantee on RSS, page cache,
or total process memory.

## Initial conditions

`hyowon_make_ic` is the supported native IC generator. It accepts only
`ic.mode="generate"`, does not run under MPI, and does not invoke CAMB, CLASS, or
another Boltzmann solver.

Generated IC coordinates include:

| TOML key | Requirement |
|---|---|
| `ic.lpt_order` | `1` or `2`. |
| `ic.seed` | Explicit `uint64`; zero is valid when written explicitly. |
| `ic.mesh_per_dimension` | `0` selects N for 1LPT and 2N for 2LPT; otherwise a multiple of N, with 2LPT requiring at least 2N. |
| `ic.max_mode_per_axis` | Optional Cartesian support K with `K>0` and `2K<N`; omission uses `floor((N-1)/2)`. |
| `ic.power_spectrum_file` | External two-column linear P(k) table. |
| `ic.power_spectrum_redshift` | Epoch represented by the table, >-1. |
| `ic.power_spectrum_fidelity` | Currently `precision_boltzmann`. |
| `ic.amplitude_mode` | `gaussian` or `fixed`. |
| `ic.phase_pairing` | `independent`, `pair_a`, or `pair_b`. |

The spectrum file contains `k [h/Mpc]` and `P(k) [(Mpc/h)^3]`, with positive
finite values and strictly increasing k. Before the first data row it must bind:

```text
# k_unit=h/Mpc
# power_unit=(Mpc/h)^3
# redshift=...
# species=cold_plus_baryon
# gauge=synchronous
# fidelity=precision_boltzmann
# normalization=sigma8_z0
# normalization_value=...
# nonlinear=false
# massive_neutrino_sum_eV=0
# h=...
# omega_m=...
# omega_b=...
# omega_lambda=...
# n_s=...
```

HYOWON checks these bindings against the TOML and hashes the spectrum input. It
does not silently renormalize the table to the configured σ8. The table must cover
the Fourier support required by the box and selected K.

For controlled particle-resolution studies, a fixed K defines a common projected
Fourier map across N. For 2LPT the final quadratic source is projected back to the
same K; this is a controlled projected 2LPT map, not the unrestricted continuous
second-order solution. See [examples/common_ic_support.md](examples/common_ic_support.md).

The generated-2LPT and T-web second-derivative operators share one
reflection-compatible even-grid Nyquist convention: a mixed derivative vanishes
when either differentiated axis is Nyquist, while diagonal second derivatives are
retained. Supported generated-IC bandlimits exclude those Nyquist planes, but the
shared definition prevents the two consumers from drifting to different discrete
operators.

Reuse the same generated IC only when all IC-defining coordinates and input bytes
are unchanged. Changing particle count, box size, seed, LPT order, IC mesh, K,
spectrum, cosmology, spectrum epoch, or start redshift requires regeneration.

## Native artifacts and compatibility

0.0.3 uses the current canonical native identities `hyowon.snapshot` and
`hyowon.restart`. The pre-stable numbered snapshot/restart/checkpoint identities
used by 0.0.2 and earlier betas are intentionally not compatibility aliases in
0.0.3. Keep 0.0.2 available when an old beta artifact must be continued or read.
Do not rewrite old evidence merely to make it appear current.

Current readers fail closed when required native semantics are missing or
incompatible. Object hashes, restart-state hashes, IC identity and build/runtime
provenance establish lineage and integrity; none of them alone certifies physical
accuracy.

## Analysis

`hyowon_analyze` consumes native particle snapshots. A minimal field-only run is:

```sh
build/hyowon_analyze SNAPSHOT.hdf5 \
  --output-directory run/analysis \
  --field-mesh 128 --bins 32 \
  --field-interlacing interlaced \
  --field-shot-noise raw \
  --field-max-k-fraction-nyquist 0.5 \
  --skip-halos
```

The analyzer supports matter power, field cross-comparison, FoF and deblended SO
halo products, HMF, two-point correlation, equilateral-shell bispectrum, shape and
spin products, progenitor linking, and tidal-web classification. Optional products
require their explicit numerical coordinates; they are not enabled by hidden
scientific defaults.

`--hmf-min-mass` and `--hmf-max-mass` are specified in Msun/h. Two-point
correlation uses the periodic cube and supports radii up to L/2. The analyzer does
not use a survey random catalog.

## Diagnostics

Optional runtime diagnostics include memory observations, force balance and
Layzer–Irvine quantities. Pure PM additionally measures the structural work defect
between the production centered-gradient force and the reported CIC
particle-position energy at synchronized KDK endpoints.

The pure-PM closure remainder is not a pure timestep-error estimator. The PM force
is not thereby asserted to be the exact gradient of the reported particle-position
energy, and TreePM is not assigned that pure-PM interpretation. Diagnostics are
measurements, not automatic scientific pass/fail thresholds.

## Distribution

Create source archives with:

```sh
cmake --build build --target package_source
```

The source package contains the simulator/analyzer source, build files, examples,
and public documentation. Development campaigns, execution receipts, generated
data, tests, CI material and local operational artifacts are not part of the 0.0.3
release source.

`--version` reports the software version plus configure-time source commit/tree
when available. Source archives without Git metadata report the source identity as
unbound; this is descriptive provenance, not a binary attestation.

See [CHANGELOG.md](CHANGELOG.md) for changes from 0.0.2.

[GPL-3.0-or-later](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) ·
[Citation metadata](CITATION.cff)
