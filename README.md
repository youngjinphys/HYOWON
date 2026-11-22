# HYOWON

HYOWON is a C++20 cosmological N-body code for collisionless structure formation
in a periodic box with a spatially flat matter–Λ background. It provides:

- Particle–Mesh (PM) gravity and an experimental TreePM solver.
- 1LPT and projected 2LPT initial conditions from an external linear spectrum.
- Cosmological kick–drift–kick evolution, HDF5 snapshots and checkpoint restart.
- CPU/OpenMP execution, optional MPI/FFTW-MPI, and snapshot analysis.

**Version 0.1.2** is a maintenance release. It improves failure handling and
input robustness without changing the numerical model. See the [changelog](CHANGELOG.md).

[GPL-3.0-or-later](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) ·
[Citation metadata](CITATION.cff) · [User guide](docs/user_guide.md)

## Requirements and build

Requires CMake 3.20+, C and C++20 compilers, FFTW3 and HDF5 C libraries, and Boost
headers for the analyzer. The default build enables OpenMP and threaded FFTW;
MPI is optional. On Debian/Ubuntu:

```sh
sudo apt install build-essential cmake git libfftw3-dev libhdf5-dev libboost-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

Configuration fetches a pinned `toml++` commit unless
`FETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS` names a local source tree.
See [build options](docs/user_guide.md#build-options) for serial builds,
installation, and
[MPI execution](docs/user_guide.md#mpi-execution) for distributed builds.

| Program | Purpose |
|---|---|
| `hyowon` | Evolve generated ICs or a native snapshot; resume checkpoints. |
| `hyowon_make_ic` | Generate a reusable native IC snapshot and optional evidence. |
| `hyowon_analyze` | Measure fields and halo properties from native snapshots. |

Each program accepts `--help` and `--version`.

## Quick start

Run commands from the repository root. **Relative paths, including paths inside
TOML files, resolve from the process working directory.**

```sh
build/hyowon examples/quickstart_generate.toml
```

This 64-particle execution example evolves from `z=9` to `z=8` and writes under
`runs/quickstart_generate/`. To repeat it, choose a fresh run label:

```sh
build/hyowon examples/quickstart_generate.toml \
  --set output.run_label=quickstart_second
```

The bundled CAMB spectrum and its small background-model mismatch are documented
in the [spectrum notes](examples/quickstart_camb/README.md).

## Initial conditions

To generate a reusable IC and evolve it separately:

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

For a custom calculation, adapt [minimal_generate.toml](examples/minimal_generate.toml)
and [minimal_snapshot.toml](examples/minimal_snapshot.toml). Match the spectrum,
cosmology, box, particle sampling and start epoch; the
[IC requirements](docs/user_guide.md#initial-condition-requirements) describe the
input header, units and Fourier support. Keep the original IC file for
[restart](docs/user_guide.md#restart) and multi-rank evolution.

## Analysis

Analyze the first output of the two-stage example:

```sh
build/hyowon_analyze run/output/minimal_snapshot/snapshots/snapshot_1.hdf5 \
  --output-directory run/analysis \
  --field-mesh 8 --bins 4 \
  --field-interlacing interlaced \
  --field-shot-noise raw \
  --field-max-k-fraction-nyquist 0.5 \
  --skip-halos
```

The output directory must be new and its parent must exist. Additional products
include field cross-comparisons, FoF candidates, spherical-overdensity measurements,
mass functions, two-point correlations, equilateral-shell bispectra, shapes,
spin, pairwise snapshot links and tidal-web classification. See
[analysis semantics](docs/user_guide.md#analysis-semantics) for their definitions.

## Scope and limitations

Pure PM is the primary Paper 0 solver; TreePM is experimental. HYOWON evolves one
collisionless matter fluid and omits hydrodynamics, baryonic feedback, radiation
evolution and massive-neutrino clustering.

Particle sampling, force mesh, IC support and timestep are independent numerical
choices. Requested output epochs terminate steps and **reset the logarithmic
cadence anchor**; changing the snapshot schedule can therefore change the
trajectory. Hold that schedule fixed when comparing other numerical choices.

Scientific accuracy requires convergence checks for the chosen observable. The
tiny examples exercise execution only. Changing FFTW planning, thread count or
MPI decomposition can change floating-point results; recorded provenance helps
identify those execution differences. The guide explains
[force resolution and scratch storage](docs/user_guide.md#numerical-setup),
[diagnostics](docs/user_guide.md#diagnostics) and
[native artifact compatibility](docs/user_guide.md#native-artifacts-and-provenance).

## Distribution and citation

See [source packaging](docs/user_guide.md#source-packaging) for archives and build
identity. Cite the metadata of the version actually used in the calculation.
**The Paper 0 release baseline is [v0.1.0](https://github.com/youngjinphys/HYOWON/releases/tag/v0.1.0).**
Version 0.1.2 does not replace the source provenance of earlier calculations.
[CITATION.cff](CITATION.cff) names Youngjin Yeom as author and records the release
version; no DOI is claimed.
