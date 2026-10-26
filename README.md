# HYOWON

`0.1.1` — the Paper 0 public research release of a C++20 cosmological N-body
implementation for a flat, periodic, collisionless, one-fluid matter–Λ model. It
provides pure PM, an experimental TreePM path, 1LPT/2LPT initial conditions,
native HDF5 snapshots and restart, optional MPI/FFTW-MPI execution, and a
standalone analysis program.

HYOWON does not model hydrodynamics, radiation evolution, baryonic feedback, or
massive-neutrino clustering.

> **Status:** public research release for Paper 0. A software version identifies
> source semantics; it does not certify cosmological convergence or general
> precision-cosmology accuracy for an arbitrary calculation.

[GPL-3.0-or-later](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) ·
[Citation metadata](CITATION.cff) · [Changelog](CHANGELOG.md)

Version 0.1.1 preserves the v0.1.0 numerical operators and artifact schemas while
reducing repeated CIC geometry work, parallelizing exact segment sums, and
streamlining SHA-256. It also makes checkpoint loading transactional through
manifest/rank agreement, closes MPI preparation error gaps, fixes locale-sensitive
text output, and refreshes source identity at build time. See the
[changelog](CHANGELOG.md#011) for scope and compatibility details.

## Requirements and build

- CMake 3.20 or newer
- C and C++20 compilers
- FFTW3
- HDF5 C library
- Boost headers for analyzer exact-integer arithmetic
- OpenMP by default
- MPI and FFTW-MPI only when distributed execution is enabled

`toml++` is fetched from a pinned upstream commit unless
`FETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS` points to a local source tree.
The first configure therefore needs network access unless that source is supplied.

Run the commands below from the repository root. Relative input and output paths
are resolved from the process working directory, including paths inside TOML
files; they are not relative to the TOML file's directory.

On Debian/Ubuntu, the default build dependencies can be installed with
`sudo apt install build-essential cmake git libfftw3-dev libhdf5-dev libboost-dev`.
The analyzer needs the Boost development headers; an existing header installation
can instead be selected with `-DHYOWON_BOOST_INCLUDE_DIR=/path/to/include`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DHYOWON_BUILD_TESTING=ON -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Optional installation (the configure flag above retains non-system dependency
search paths for installed executables; those libraries must remain available):

```sh
cmake --install build --prefix "$HOME/.local"
```

Useful options are `HYOWON_ENABLE_OPENMP`, `HYOWON_ENABLE_FFTW_THREADS`,
`HYOWON_ENABLE_MPI`, `HYOWON_ENABLE_FFTW_MPI`, `HYOWON_BUILD_ANALYZER`,
`HYOWON_OPENMP_ROOT`, `HYOWON_FFTW_ROOT`, and `HYOWON_BOOST_INCLUDE_DIR`.
MPI and FFTW-MPI must be enabled together for distributed FFT execution.
HYOWON-owned C++ targets disable floating-point reassociation and unsafe
fast-math transformations because exact boundary predicates and deterministic
reductions rely on ordinary IEEE-754 evaluation order.

For a build with OpenMP and FFTW threading disabled:

```sh
cmake -S . -B build-serial -DCMAKE_BUILD_TYPE=Release \
  -DHYOWON_ENABLE_OPENMP=OFF \
  -DHYOWON_ENABLE_FFTW_THREADS=OFF
cmake --build build-serial --parallel 2
```

Use `build-serial/` instead of `build/` in the execution examples for that build.
The default build enables OpenMP/FFTW threading and has MPI disabled. For an
OpenMP-enabled build, set `runtime.num_threads` in the TOML file (or pass
`--set runtime.num_threads=2` to `hyowon`) to choose the host thread count.
Use `runtime.num_threads=1` with `build-serial`.

`HYOWON_BUILD_TESTING=ON` enables a smoke test that generates 64-particle ICs,
evolves them, resumes a checkpoint with parent lineage, checks refusal to
overwrite existing outputs, and runs field analysis when enabled. It uses a
private temporary directory inside the build tree, removes it on success,
and retains it on failure.
This checks a bounded software path, not scientific convergence or multi-rank MPI.

## Quick start

The repository includes a tiny executable example with a bundled CAMB-backed
spectrum:

```sh
build/hyowon examples/quickstart_generate.toml
```

It evolves 64 particles from z=9 to z=8 and writes snapshots/checkpoints below
`runs/quickstart_generate/`. Existing run directories are refused. To repeat it,
choose a new label, for example `--set output.run_label=quickstart_second`.
This is a software-path example, not a converged
cosmological calculation. Generator provenance and the small background-model
mismatch are documented in [the quick-start spectrum notes](examples/quickstart_camb/README.md).

To generate a reusable IC and evolve it separately with the same tiny example:

```sh
mkdir -m 700 run
cp examples/quickstart_generate.toml run/generate.toml
cp examples/minimal_snapshot.toml run/snapshot.toml

build/hyowon_make_ic run/generate.toml \
  --output run/ic.hdf5 --evidence run/ic.json

build/hyowon run/snapshot.toml \
  --set ic.snapshot_file=run/ic.hdf5 \
  --set output.root_directory=run/output
```

For a custom calculation, start from `examples/minimal_generate.toml` and
`examples/minimal_snapshot.toml`. Set `ic.power_spectrum_file` and
`ic.power_spectrum_redshift` in the generate config to a matching external table.
Set `[cosmology]`, box size, particle count and start redshift consistently in
both configs; snapshot ingestion checks them against the IC. The maintained
background requires `omega_m + omega_lambda = 1`. Set strictly increasing
`snapshot_scale_factors` after the start scale factor and at or before the final
scale factor. See the spectrum requirements below.

Keep the original IC file. Restart reopens it to verify the bound source identity.
Resume with the original cosmology, box/force settings, IC identity, time interval,
timestep, snapshot schedule and MPI mode/rank count. Use a fresh output directory
or label. For the two-stage example above:

```sh
build/hyowon run/snapshot.toml \
  --restart run/output/minimal_snapshot/checkpoints/restart_step_3 \
  --set ic.snapshot_file=run/ic.hdf5 \
  --set output.root_directory=run/output \
  --set output.run_label=resumed
```

A multi-rank PM launch needs MPI C++ development tools/runtime and FFTW-MPI
headers/libraries in addition to the default dependencies. On Debian/Ubuntu,
the Open MPI packages are `libopenmpi-dev`, `openmpi-bin` and `libfftw3-mpi-dev`.
Configure both MPI options in a separate build directory, then launch the
snapshot example with MPI enabled and a fresh output label. The prefixes below
select the Debian/Ubuntu system packages; replace them with the intended HDF5
and FFTW prefixes for other installations.

Use one coherent FFTW installation for the base, threaded and MPI libraries,
compatible with the selected MPI runtime. If multiple installations coexist,
use a fresh build directory with explicit prefixes. The runtime loader must
select the intended libraries; resolve CMake runtime-search-path conflicts
before launching. On Linux, inspect `ldd build-mpi/hyowon` after building: the
base, threaded and MPI FFTW libraries must resolve to the intended installation.
For a custom MPI installation, select `MPI_CXX_COMPILER` at configure time and
use its matching `mpiexec`.

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

Distributed PM evolution uses FFTW-MPI rank-local one-dimensional slabs. Native
IC generation remains serial; fresh multi-rank evolution starts from an immutable
generated snapshot rather than a distributed LPT pipeline.

## Scope and limitations

`box.particles_per_dimension=N` defines the particle sampling and
`box.pm_mesh_per_dimension` defines an independent force-mesh coordinate. Pure PM
is the primary Paper 0 path; TreePM remains experimental. The maintained
integrator uses one synchronized global `delta_ln_a` cadence, with requested
output epochs inserted as exact step boundaries. Halo-core or other short-time
claims therefore require explicit timestep refinement.

`gravity.deconvolve_cic` is an explicit force-method coordinate, not a universal
accuracy switch. A finer PM mesh at fixed particle sampling is likewise not
automatically more accurate: localized mass assignment, particle-lattice phase,
aliasing, derivative kernels, and deconvolution can interact. The bundled examples
use `deconvolve_cic=false`. HYOWON does not globally reject other represented
settings merely because a bounded experiment performs poorly; such settings
require observable-specific refinement before precision claims are made.

Changing FFTW thread count, planner state, MPI decomposition, or another execution
path is not promised to be bitwise invariant. Planning/runtime provenance is
recorded so execution-path sensitivity can be measured rather than silently
identified with physical convergence.

Scratch mode defaults to memory. Disk scratch changes backing placement only and
is not a guarantee on RSS or page-cache use. Prefer node-local scratch over a
shared filesystem when available and benchmark the selected environment.

Scientific claims require independent refinement of the numerical coordinates
relevant to the target observable, including start redshift, timestep, force mesh,
particle sampling, box size, and IC support.

## Initial conditions

`hyowon_make_ic` supports native generated 1LPT/2LPT ICs from an external linear
cold-plus-baryon power spectrum. It does not invoke CAMB or CLASS internally and
does not silently renormalize the supplied table to the configured σ8.

Generated IC coordinates include:

| TOML key | Requirement |
|---|---|
| `ic.lpt_order` | `1` or `2`. |
| `ic.seed` | Explicit `uint64`; zero is valid. |
| `ic.mesh_per_dimension` | `0` selects the automatic policy. Explicit M is a positive integer multiple of N. |
| `ic.max_mode_per_axis` | Optional Cartesian support K with `K>0` and `2K<N`; omission uses `floor((N-1)/2)`. |
| `ic.power_spectrum_file` | External two-column linear P(k) table. |
| `ic.power_spectrum_redshift` | Epoch represented by the table, >-1. |
| `ic.power_spectrum_fidelity` | Currently `precision_boltzmann`. |
| `ic.amplitude_mode` | `gaussian` or `fixed`. |
| `ic.phase_pairing` | `independent`, `pair_a`, or `pair_b`. |

The spectrum uses `k [h/Mpc]` and `P(k) [(Mpc/h)^3]`, positive finite values, and
strictly increasing k. The required `# key=value` header syntax is illustrated by
the first 15 lines of [the bundled table](examples/quickstart_camb/linear_cb_z0.dat).
Accepted semantics are synchronous-gauge `cold_plus_baryon`,
`fidelity=precision_boltzmann`, `normalization=sigma8_z0`, `nonlinear=false` and
zero massive-neutrino mass. Header cosmology, normalization value and spectrum
redshift must match the configuration and the actual supplied spectrum; editing
metadata alone does not make a different spectrum compatible. Input bytes are hashed.
The table must cover the Fourier modes requested by the selected box and support;
out-of-range spectrum queries are errors.

### Projected 2LPT support

Generated density uses the inclusive Cartesian band
`|m_x|,|m_y|,|m_z| <= K`. The quadratic 2LPT source is projected back to the same
band after its source FFT. The retained projected band therefore requires
`M > 3K`; this does not claim that the complete `[-2K,2K]` quadratic convolution
is represented alias-free, which would require `M > 4K` without Nyquist
identification.

The independent condition `M % N == 0` comes from direct IC-mesh-node sampling of
the particle lattice. With automatic IC mesh selection, 1LPT uses `M=N`; projected
2LPT uses `M=N` when `N > 3K`, otherwise `M=2N`. See
`examples/common_ic_support.md` for the retained-band derivation.

Generated LPT uses the corner lattice `q_i = i L/N`, persisted as
`ic_lattice_convention="corner"`. Its phase relative to a chosen evolution PM mesh
is therefore part of the represented numerical setup. HYOWON does not currently
phase-average that relation or claim PM-force accuracy independent of it. In
matched force-resolution experiments, keep the generated IC fixed rather than
shifting the initial lattice as a function of the PM mesh; otherwise the force
intervention and the initial condition are confounded.

Generated 2LPT and T-web share one reflection-compatible even-grid Hessian
convention: mixed derivatives vanish when either differentiated axis is Nyquist,
while diagonal second derivatives are retained. Supported generated-IC bands
exclude those Nyquist planes.

Reuse a generated IC only when all IC-defining coordinates and input bytes are
unchanged. Changing particle count, box size, seed, LPT order, IC mesh, K,
spectrum, cosmology, spectrum epoch, or start redshift requires regeneration.

## Native artifacts and compatibility

The canonical native snapshot and restart identities are `hyowon.snapshot` and
`hyowon.restart`. Earlier numbered pre-release identities are historical contracts,
not aliases to be renamed. Current readers fail closed when required native
semantics are missing or incompatible. Object hashes, restart-state hashes, IC
identity, and build/runtime provenance establish lineage and integrity; none alone
certifies physical accuracy.

A newer reader/writer contract can be asymmetric with an older pre-release. Do not
rewrite artifact metadata to force an older release to accept a state outside its
declared semantics.

## Analysis

`hyowon_analyze` supports matter power, field cross-comparison, FoF and deblended
SO products, HMF, two-point correlation, equilateral-shell bispectrum, shape and
spin products, progenitor linking, and tidal-web classification. Optional products
require explicit numerical coordinates.

The current HMF and two-point-correlation products are statistics of the retained
FoF-candidate catalogue, not named SO rows. Named `M200m`, `M200c`, and
`Mvir_BN98` products are measurements around retained density-peak seeds; their
apertures may overlap and they do not by themselves form a distinct-host/subhalo
catalogue.

FoF catalogue centers use the intrinsic mass-weighted Fréchet center of represented
periodic particle coordinates; this need not coincide with a density peak,
potential minimum or most-bound particle.

Closed-aperture and FoF membership compare the exact torus squared distance of the
represented canonical binary64 endpoints with the requested radius. Fast outward-
rounded filters may certify a relation; ambiguous cases fall back to exact integer
arithmetic. Rounded displacement/radius values are descriptive quantities rather
than boundary predicates.

Named SO shell ordering and shell membership use that exact represented-endpoint
geometry, but the mean-density crossing radius is a binary64 projection of the
once-rounded enclosed mass and represented density parameters. Exact membership
predicates do not make that continuous density root arbitrary precision.

A minimal field-only analysis of the two-stage example is shown below. Its output
directory must be new and its parent must exist (`run/` was created above).
The small mesh and bin count exercise the software path only.

```sh
build/hyowon_analyze run/output/minimal_snapshot/snapshots/snapshot_1.hdf5 \
  --output-directory run/analysis \
  --field-mesh 8 --bins 4 \
  --field-interlacing interlaced \
  --field-shot-noise raw \
  --field-max-k-fraction-nyquist 0.5 \
  --skip-halos
```

## Diagnostics

Optional diagnostics include memory observations, force balance, and Layzer–Irvine
quantities. Pure PM additionally measures the structural work defect between the
production centered-gradient force and the reported CIC particle-position energy at
synchronized KDK endpoints.

The pure-PM Layzer–Irvine closure remainder is not a pure timestep-error estimator.
The PM force is not thereby asserted to be the exact coordinate gradient of the
reported particle-position energy. Diagnostics are measurements, not automatic
scientific pass/fail thresholds.

## Distribution

Create source archives with:

```sh
cmake --build build --target package_source
```

Source packages contain the maintained simulator/analyzer source, build files,
examples, license, citation metadata, and release documentation. Produce formal
assets only from a clean checkout of the exact release commit/tag. Archive SHA-256
sidecars identify archive bytes; they are not scientific validation results.

`--version` reports the software version/release stage and build-time source
commit/tree/state when Git metadata is available. The header is refreshed before
executable compilation; unchanged observations do not trigger recompilation.
Commit/tree identify Git HEAD, and a dirty state does not identify the exact
uncommitted contents or attest the executable. Source archives without Git
metadata report an unbound source identity.

See [CHANGELOG.md](CHANGELOG.md) for release changes.

## Citation

If you use HYOWON in research, cite the software using [CITATION.cff](CITATION.cff).
The current metadata identifies Youngjin Yeom as the author and version `0.1.1`;
it does not declare a DOI. Prefer the metadata of a tagged release when citing
published work.
