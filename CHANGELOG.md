# Changelog

## 0.1.0 — Paper 0 release

First Paper 0 public research release. Changes are compared with public HYOWON
`v0.0.3` at `125edabfd7dcdbb17b9a77f09ac34a4cbdccca6b`.

### Numerical semantics

- Replace the proxy projected-2LPT mesh condition `M >= 2N` with the retained-band
  condition `M > 3K`. The quadratic source is projected back to the inclusive
  Cartesian support `|m_i| <= K`; preserving the complete `[-2K,2K]` convolution
  would be a different problem. The independent `M % N == 0` condition remains a
  direct particle-lattice sampling constraint.
- Keep the historical automatic `M=2N` choice for default near-particle-Nyquist
  support while admitting `M=N` for deliberately reduced support when `N > 3K`.
- Preserve exact represented-binary64 periodic membership and ordering semantics.
  Fast outward-rounded interval filters may certify distance/radius relations;
  ambiguous cases fall back to exact dyadic integer arithmetic. Exact failures are
  propagated rather than converted into an `outside` result.
- Use monotone correctly-rounded squared-distance keys to avoid repeating exact
  arithmetic during halo distance sorting. Equal keys remain ambiguous and are
  resolved with the exact comparator.
- Group SO and Vmax particles by exact represented periodic shells. Vmax projects
  the exact shell distance to nearest-even binary64 with an exact midpoint decision
  and carries its exact positive enclosed-mass accumulator into the logarithmic
  circular-speed calculation.
- Replace the previous circular periodic FoF mass center with the intrinsic
  mass-weighted Fréchet center on the represented three-torus. This is a precise
  catalogue center-of-mass estimator, not a universal physical halo-center claim.
- Admit the TreePM `theta=0` exact-opening endpoint while retaining the configured
  short-range cutoff, softening, and leaf arithmetic.
- Admit the two-knot natural-cubic limit and evaluate the resulting log-linear
  affine quotient from exact represented dyadic inputs with one nearest-even
  binary64 rounding.
- Define the Layzer-Irvine normalized ratio directly when its denominator is
  nonzero. The pure-PM closure remainder remains a diagnostic rather than a pure
  timestep-error estimator.

### Runtime and reproducibility

- Add an opt-in `HYOWON_BUILD_TESTING` CTest smoke check covering bundled IC
  generation, evolution, restart lineage, existing-output protection, and field
  analysis, with isolated temporary artifacts. Document default build dependencies.
- Reuse otherwise-dead serial/replicated PM complex-FFT storage as transient
  deterministic CIC ordering scratch when capacity permits; distributed or small
  buffers retain the owned-scratch path.
- Parallelize the serial/replicated deposited-mass audit with exact mergeable
  positive sums without changing the represented final mass.
- Keep distributed PM on FFTW-MPI rank-local slabs. Native generated LPT remains a
  serial capability and distributed runs start from generated snapshots.
- Record FFTW planning/runtime provenance. Bitwise identity across different thread
  counts, planner states, or MPI decompositions is not a release contract.

### Paper 0 interpretation

- Generated LPT uses the persisted corner-lattice convention `q_i=iL/N`. Its phase
  relative to the evolution PM mesh is part of the represented numerical setup;
  matched force-resolution comparisons must keep the generated IC fixed.
- `gravity.deconvolve_cic` and PM mesh size remain independent numerical choices.
  No mesh/deconvolution combination is globally promoted to a precision setting
  from a bounded single-mode result.
- Keep named SO products as density-peak-seeded measurements that may overlap and
  do not by themselves define a distinct-host/subhalo catalogue. HMF and 2PCF
  remain statistics of the retained FoF-candidate catalogue.
- Keep source lineage, software-path execution, numerical response experiments,
  cross-code comparisons, and physical convergence as distinct evidence classes.
  An external code used for comparison is not designated as ground truth.

### Distribution and scope

- Keep the tracked release tree limited to maintained source, build files, examples,
  release metadata, licensing, and citation material; generated build/simulation
  artifacts are excluded from source archives.
- Keep a small CAMB-backed quick-start spectrum with CAMB version, parameters, and
  sanitized generation provenance so the bundled example is executable without a
  Boltzmann-code installation.
- Source archives use `HYOWON-v0.1.0` versioned names and SHA-256 sidecars.
- Native artifact identities remain `hyowon.snapshot` and `hyowon.restart`; earlier
  numbered pre-release identities remain historical contracts.
- Pure PM is the primary Paper 0 path; TreePM remains experimental. No
  hydrodynamics, radiation evolution, baryonic feedback, massive-neutrino
  clustering, or general precision-cosmology certificate is added by this release.

## 0.0.3 — Beta

Public release commit: `125edabfd7dcdbb17b9a77f09ac34a4cbdccca6b`.

### Runtime, analysis, and numerical integrity

- Correct HMF user-unit conversion, periodic endpoint handling, collective MPI
  startup/error synchronization, disk-scratch failure handling, and analysis-only
  build separation.
- Disable unsafe floating-point reassociation for HYOWON C++ targets so exact
  boundary predicates and compensated reductions are not silently changed by
  compiler fast-math semantics.
- Add pure-PM force-energy work diagnostics at synchronized KDK endpoints without
  changing the production PM force operator.

### Initial conditions and provenance

- Bind generated-IC source configuration and Fourier-support provenance more
  strictly, protect source inputs from output-path aliasing, and preserve the shared
  reflection-compatible Hessian convention used by generated 2LPT and T-web.
- Keep generated ICs based on explicit external precision-Boltzmann spectra; the
  generator does not invoke a Boltzmann solver or silently renormalize P(k).

### Native artifacts and release surface

- Introduce the canonical versionless native identities `hyowon.snapshot` and
  `hyowon.restart`. Numbered beta identities from 0.0.2 and earlier are not treated
  as compatibility aliases.
- Keep the distributed source limited to maintained release files; local development
  artifacts are not shipped.
- Add `CITATION.cff` and keep a compact README, examples, license, and third-party
  notices.

### Scope

- 0.0.3 is a Beta research implementation. Its version label is not a universal
  precision-cosmology, large-volume, multi-node, or convergence certificate.

## 0.0.2 — Beta

Compared with v0.0.1, commit `04344e0376989a63486e206665a6373c48386836`.

- Bind requested IC Fourier support into restart and MPI numerical identities.
- Bound MPI snapshot verification buffers and synchronize rank-local runtime setup
  and validation publication errors before later collectives.
- Preserve coordinate-reflection symmetry in T-web classification on even meshes
  and share the Fourier Hessian implementation with LPT.
- Advance build and CLI identity to 0.0.2 while retaining Beta status.

Numbered beta snapshot/restart identities used by this release are intentionally
not the current canonical artifact identities.

## 0.0.1 — Beta

First public beta release.
