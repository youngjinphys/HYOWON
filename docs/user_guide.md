# HYOWON user guide

The [README](../README.md) covers the model, build, quick start, separate IC
generation and field analysis. This guide records the additional usage instructions
and numerical definitions. All example commands run from the repository
root; paths inside TOML files use the process working directory.

## Build options

The default build enables OpenMP/FFTW threading and disables MPI. Useful CMake
options are `HYOWON_ENABLE_OPENMP`, `HYOWON_ENABLE_FFTW_THREADS`,
`HYOWON_ENABLE_MPI`, `HYOWON_ENABLE_FFTW_MPI` and `HYOWON_BUILD_ANALYZER`.
`HYOWON_OPENMP_ROOT`, `HYOWON_FFTW_ROOT` and `HYOWON_BOOST_INCLUDE_DIR` select
dependency installations. The analyzer uses Boost headers for exact-integer
arithmetic. On macOS, install `libomp` or set `HYOWON_OPENMP_ROOT` for OpenMP.

For a serial build:

```sh
cmake -S . -B build-serial -DCMAKE_BUILD_TYPE=Release \
  -DHYOWON_ENABLE_OPENMP=OFF -DHYOWON_ENABLE_FFTW_THREADS=OFF
cmake --build build-serial --parallel 2
```

Use `build-serial/` in place of `build/` in the examples and set
`runtime.num_threads=1`. With OpenMP, select the thread count in TOML or pass
`--set runtime.num_threads=2` to `hyowon`.

HYOWON C++ targets disable unsafe fast-math transformations and floating-point
reassociation: exact boundary predicates and deterministic reductions depend on
ordinary IEEE-754 evaluation order.

### Installation

To retain search paths for dependencies installed outside system locations:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
cmake --build build --parallel 2
cmake --install build --prefix "$HOME/.local"
```

The selected dependency libraries must remain available after installation.

## MPI execution

Distributed FFT execution requires both `HYOWON_ENABLE_MPI=ON` and
`HYOWON_ENABLE_FFTW_MPI=ON`. On Debian/Ubuntu, install `libopenmpi-dev`,
`openmpi-bin` and `libfftw3-mpi-dev` in addition to the default dependencies.

Use one coherent FFTW installation for its base, threaded and MPI libraries,
compatible with the selected MPI runtime. If several installations coexist,
configure a fresh build directory with explicit prefixes. Resolve CMake runtime
search-path conflicts before launching. On Linux, `ldd build-mpi/hyowon` shows
which libraries will load. For a custom MPI installation, select
`MPI_CXX_COMPILER` and use its matching `mpiexec`.

After generating `run/ic.hdf5` with the README's two-stage example:

```sh
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release \
  -DHYOWON_ENABLE_MPI=ON -DHYOWON_ENABLE_FFTW_MPI=ON \
  -DHDF5_ROOT=/usr -DHYOWON_FFTW_ROOT=/usr
cmake --build build-mpi --parallel 2

mpiexec -n 2 build-mpi/hyowon run/snapshot.toml \
  --set ic.snapshot_file=run/ic.hdf5 \
  --set runtime.mpi_enabled=true \
  --set output.root_directory=run/output \
  --set output.run_label=mpi
```

Replace `/usr` with the intended dependency prefixes outside Debian/Ubuntu
system installations. Distributed PM uses FFTW-MPI rank-local one-dimensional
slabs. Native LPT generation is serial, so fresh multi-rank runs begin from an
existing native IC snapshot.

## Restart

Keep the original IC file: restart reopens it to verify the bound source
identity. Preserve the original cosmology, box/force settings, IC identity, time
interval, timestep, snapshot schedule and MPI mode/rank count. Select a fresh
output directory or run label.

For the README's two-stage example:

```sh
build/hyowon run/snapshot.toml \
  --restart run/output/minimal_snapshot/checkpoints/restart_step_3 \
  --set ic.snapshot_file=run/ic.hdf5 \
  --set output.root_directory=run/output \
  --set output.run_label=resumed
```

## Initial-condition requirements

`hyowon_make_ic` generates 1LPT or projected 2LPT ICs from an external linear
cold-plus-baryon spectrum. It does not invoke CAMB or CLASS or renormalize the
table to the configured σ8.

| TOML key | Requirement |
|---|---|
| `ic.lpt_order` | `1` or `2`. |
| `ic.seed` | Explicit non-negative TOML integer, `0` through `9223372036854775807` (`INT64_MAX`). |
| `ic.mesh_per_dimension` | `0` selects the automatic policy; explicit M is a positive integer multiple of particle dimension N. |
| `ic.max_mode_per_axis` | Optional Cartesian support K, with `K>0` and `2K<N`; omission uses `floor((N-1)/2)`. |
| `ic.power_spectrum_file` | External two-column linear P(k) table. |
| `ic.power_spectrum_redshift` | Epoch represented by the table, greater than −1. |
| `ic.power_spectrum_fidelity` | `precision_boltzmann`. |
| `ic.amplitude_mode` | `gaussian` or `fixed`. |
| `ic.phase_pairing` | `independent`, `pair_a` or `pair_b`. |

The spectrum uses `k [h/Mpc]` and `P(k) [(Mpc/h)^3]`, positive finite values, and
strictly increasing k. The required `# key=value` header is illustrated in the
[bundled table](../examples/quickstart_camb/linear_cb_z0.dat). Accepted semantics
are synchronous-gauge `cold_plus_baryon`, `fidelity=precision_boltzmann`,
`normalization=sigma8_z0`, `nonlinear=false` and zero massive-neutrino mass.
Header cosmology, normalization value and spectrum redshift must match the
configuration and the actual spectrum. Input bytes are hashed; editing metadata
alone cannot make a different spectrum compatible.

The table must cover the requested Fourier modes; out-of-range queries are errors.
The σ8 diagnostic integrates over the supplied table support and reports its
difference from the declared normalization without renormalizing the table.
Match cosmology, box, particle sampling and start redshift between generation and
snapshot configurations. The background
requires `omega_m + omega_lambda = 1`. Snapshot scale factors must be strictly
increasing, after the start epoch and at or before the final epoch.

### Projected 2LPT and lattice sampling

The generated density uses the inclusive Cartesian band
`|m_x|,|m_y|,|m_z| <= K`. After its FFT, the quadratic 2LPT source is projected
back to that same band. Avoiding quadratic aliases in the retained band requires
`M > 3K`; representing the complete `[-2K,2K]` convolution without identifying
Nyquist edge modes would require `M > 4K`.

The separate condition `M % N == 0` comes from direct IC-mesh-node sampling of
the particle lattice. Automatic selection uses `M=N` for 1LPT; for projected
2LPT it uses `M=N` when `N > 3K`, otherwise `M=2N`. See
[common IC support](../examples/common_ic_support.md) for the derivation.

LPT uses the corner lattice `q_i = i L/N`, recorded as
`ic_lattice_convention="corner"`. Its phase relative to the evolution mesh is
part of the numerical setup. For force-resolution comparisons, keep the
generated IC fixed; shifting its lattice with the force mesh changes both.

Generated 2LPT and T-web use the same reflection-compatible even-grid Hessian:
mixed derivatives vanish when either differentiated axis is Nyquist; diagonal
second derivatives remain. Supported generated-IC bands exclude those planes.

Regenerate ICs when particle count, box size, seed, LPT order, IC mesh, K,
spectrum bytes, cosmology, spectrum epoch, start redshift, amplitude mode or
phase pairing changes.

## Numerical setup

`box.particles_per_dimension=N` sets particle sampling;
`box.pm_mesh_per_dimension` sets the independent force mesh. The global KDK
cadence and its dependence on output times are described in the
[README](../README.md#scope-and-limitations). Short-time or halo-core studies
require timestep refinement as well as spatial refinement.

`gravity.deconvolve_cic` selects a force method. A finer mesh or enabling
deconvolution does not automatically improve accuracy at fixed particle
sampling: assignment, lattice phase, aliasing and derivative kernels interact.
The bundled examples use `deconvolve_cic=false`. Assess the selected settings
against the observable and independently vary start redshift, timestep, force
mesh, particle sampling, box size and IC support as appropriate.

Scratch defaults to memory. Disk mode changes backing placement while preserving
stored values; it does not impose an RSS or page-cache limit. Prefer node-local
scratch when available and measure the selected filesystem's behavior.

## Analysis semantics

Optional analysis products require explicit numerical settings; run
`build/hyowon_analyze --help` for the options.

- HMF and two-point-correlation products describe the retained FoF-candidate
  catalogue. Named `M200m`, `M200c` and `Mvir_BN98` measurements use retained
  density-peak seeds; their apertures may overlap and do not define a complete
  distinct-host/subhalo catalogue.
- FoF centers use the intrinsic mass-weighted Fréchet mean on the periodic
  domain. This center can differ from a density peak or potential minimum.
- FoF and closed-aperture membership compare exact torus squared distances
  between represented canonical binary64 endpoints; ambiguous floating-point filters
  fall back to exact integer arithmetic. SO shell ordering uses that geometry,
  while the mean-density crossing radius remains a binary64 result. Exact
  membership therefore does not imply arbitrary-precision radii or physical
  convergence.
- Pairwise particle-ID overlap links connect two snapshots; they are not a
  full merger tree. Equilateral-shell bispectra require uniform-mass particles.
- Power-spectrum interlacing and CIC-aliased Poisson shot-noise subtraction
  cannot currently be combined. Use raw power with interlacing, or select the
  single-grid estimator for that subtraction.

## Diagnostics

Optional diagnostics report memory use, force balance and Layzer–Irvine
quantities. Pure PM also measures the structural work defect between its
centered-gradient force and the reported CIC particle-position energy at
synchronized KDK endpoints. That force need not be the coordinate gradient of
the reported energy, so the Layzer–Irvine closure remainder includes more than
timestep error. These measurements carry no automatic scientific pass/fail
threshold.

## Native artifacts and provenance

Native snapshot and restart identities are `hyowon.snapshot` and
`hyowon.restart`. Earlier numbered pre-release formats have distinct contracts;
changing their metadata does not make them compatible. Readers reject missing
or incompatible required semantics. Newer/older reader compatibility can be
asymmetric.

Object hashes, restart-state hashes, IC identity and build/runtime metadata
record integrity and lineage. `--version` reports build-time Git commit, tree
and state when available; the header refreshes before executable compilation.
Commit/tree describe HEAD. A dirty state neither identifies the exact local
edits nor attests the executable. Source archives without Git metadata report
an unbound source identity.

### FFTW wisdom cache

Serial FFTW plans start without persisted wisdom unless
`HYOWON_FFTW_WISDOM_DIRECTORY` names an existing absolute, non-symlink directory.
An explicit cache is create-once: existing files are never overwritten. New
names distinguish the initialized planner configuration as well as provider,
system, dimensions, thread count and alignment. This lets the IC generator and
simulator share a directory even when their planner configurations differ.

A compatible v0.1.1 cache takes precedence to preserve its existing plan
selection. A recognized legacy file for another planner configuration stays
untouched while the current configuration uses its own name. Malformed legacy
headers and rejected selected wisdom remain errors. Changing cache contents,
planning context or execution settings can change floating-point results;
retain the recorded plan and wisdom identities for matched comparisons.

## Source packaging

```sh
cmake --build build --target package_source
```

CPack creates source TGZ/ZIP archives and SHA-256 sidecars. The archives contain
the simulator/analyzer source, build files, examples, license, citation metadata
and documentation. Produce published assets from a clean checkout of the exact
release commit/tag. Sidecars identify archive bytes; cite the version actually
used, as explained in [the README](../README.md#distribution-and-citation).
