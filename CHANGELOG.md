# Changelog

## 0.0.3 — Beta

Compared with the public HYOWON `v0.0.2` branch at commit
`e12a1b8bb960d970aa2c70d63255b56aed46ce70`.

### Runtime and analysis correctness

- Correct `--hmf-min-mass` and `--hmf-max-mass` at the analyzer boundary. The
  documented user unit is Msun/h and is now converted exactly once to HYOWON's
  native `1e10 Msun/h` halo-mass unit before selection and binning.
- Preserve periodic endpoint separation in FoF and periodic two-point counting.
  Boundary-crossing pairs no longer lose a small represented separation by
  subtracting wrapped endpoints before applying the minimum-image rule.
- Contain CIC deposition failures outside OpenMP parallel regions instead of
  allowing C++ exceptions to escape through an OpenMP worksharing region.
- Synchronize distributed snapshot-IC owner-population allocation failures before
  later MPI collectives, and synchronize rank-local PM/conservation diagnostic
  validation, conversion and allocation failures before reductions or spectral
  communication. Empty-rank uniform-mass handling and nonrepresentable intermediate
  values now fail closed.
- Add collective MPI startup preflight for the exact configuration bytes,
  CLI/restart identity and `runtime.mpi_enabled` intent before rank-local simulation
  state is constructed. Multi-rank launches with MPI disabled are rejected, and an
  unexpected post-preflight local startup failure aborts the communicator rather
  than leaving peers blocked.
- Disable floating-point reassociation/fast-math semantics for HYOWON C++ targets.
  Compensated reductions and exact-boundary predicates no longer depend on compiler
  optimization modes that may silently change finite results.
- Harden the exact-binary64 product-sum fallback used by cancellation-sensitive
  analysis. Factor admission inspects the represented IEEE-754 value directly and
  does not allow an earlier zero factor to hide a later NaN or infinity.
- Use an AppleClang-compatible lifetime start for FFTW complex-array scratch
  storage, avoiding the array pseudo-destructor compilation failure seen with the
  previous construction path.
- Harden disk-scratch lifetime and failure handling. Failed unlink/mmap setup is
  reported instead of ignored, descriptors are released on failure paths, and a
  successful mapping no longer retains an unnecessary open descriptor for each
  live scratch buffer.
- Record the requested evolution scratch placement separately from IC scratch
  placement in runtime metadata. Snapshot-only analysis leaves unavailable
  upstream trajectory policy unset instead of inventing it.

### Initial conditions and provenance

- `hyowon_make_ic` now binds the exact source-configuration SHA-256 into its
  evidence and refuses output/evidence paths that alias the source configuration
  or linear power-spectrum input, including existing filesystem-object aliases.
  `--overwrite` therefore cannot authorize destruction of the inputs defining the
  generated IC.
- Current native generator provenance requires the IC Fourier-support fields to be
  present with their current semantics. An explicitly omitted requested bound is
  distinct from missing provenance, and unavailable effective support is rejected
  when the artifact is used as a verified generated IC.

### Diagnostics and numerical interpretation

- Add a pure-PM force-energy work diagnostic at synchronized KDK endpoints. It
  measures the directional derivative of the reported CIC particle-position energy
  together with the production centered-gradient force and integrates their signed
  structural work alongside the Layzer-Irvine timeline. The production PM force is
  unchanged.
- Preserve cancellation corrections when validating integrated force-energy work
  and Layzer-Irvine closure, avoiding an intermediate scalar rounding that could
  reject a represented small closure remainder.
- The force-energy closure remainder is a diagnostic, not a pure timestep-error
  estimator or a physical-accuracy certificate. TreePM is not assigned the pure-PM
  energy-gradient interpretation.

### Native artifact contract

- Converge the pre-stable native snapshot, restart, checkpoint, diagnostic and
  provenance families onto the current canonical, versionless contract. Public
  snapshot and restart identities are now `hyowon.snapshot` and `hyowon.restart`.
- This is an intentional **beta compatibility break**. Numbered native artifact
  identities from 0.0.2 and earlier betas are not treated as compatibility aliases
  by 0.0.3. Use the release that produced an old beta artifact when it must be
  continued or interpreted; do not relabel old evidence as current-format data.
- Keep IC Fourier-support semantics bound to the current snapshot/restart and MPI
  numerical identities. Missing required current-format semantics fail closed.

### Code and public release surface

- Remove the unused FoF-catalog reader, the unused explicit-random 2PCF overload and
  cross counter, and an allocating exchange-sort wrapper. The live periodic 2PCF
  estimator, durable FoF catalog writer and in-place exchange paths remain.
- Build analysis-only snapshot/catalog adapters only with `hyowon_analysis` when the
  analyzer is enabled instead of compiling them into the simulation core. Candidate
  halo-center storage is now created only when the corresponding 2PCF product is
  requested.
- Remove the 0.0.2 opt-in regression-test source tree from the distributed public
  source surface. No runtime behavior depends on those test executables.
- Consolidate release-facing guidance into one self-contained README plus the common
  projected-IC support note required to interpret `ic.max_mode_per_axis`.
- Add `CITATION.cff` to the public tree and retain the license and third-party notices.
- Keep generated data, execution receipts, CI/development material and local
  operational artifacts outside source archives and installation.

The items above are the public `v0.0.2` to 0.0.3 source delta. Private Paper-0
campaign tools, internal QA/review ledgers and other development-only material were
removed while preparing the candidate branch, but they were not present in the
public `v0.0.2` tree and therefore are not listed as public-release changes here.

### Pre-release candidate audit corrections

These corrections close branch-lineage regressions found while auditing the 0.0.3
candidate. They preserve or re-establish intended semantics before release; they are
not claims that the public `v0.0.2` release had these defects.

- Preserve the v0.0.2 distributed-snapshot memory bound by canonicalizing the
  effective transfer batch after rank topology is known. The effective batch is
  capped by the configured upper bound, the MPI count limit and the largest actual
  rank payload. Transport staging and post-write native reread verification now use
  that same bound, so a large configured batch cannot recreate oversized validation
  buffers for a small distributed snapshot.
- Restore one shared reflection-compatible Fourier Hessian operator for generated
  2LPT and T-web analysis. Mixed derivatives vanish when either differentiated axis
  is an even-grid Nyquist axis, including the double-Nyquist case; diagonal second
  derivatives remain. Supported generated-IC bandlimits already exclude these
  Nyquist planes, so the correction removes a latent operator divergence without
  changing supported generated trajectories.

### Numerical scope

- Pure PM remains the primary release target. TreePM is present but experimental
  and requires application-specific qualification.
- 0.0.3 remains a Beta research implementation. The version does not constitute a
  universal precision-cosmology, large-volume, multi-node or convergence
  certificate.
- Production force, IC-generation and KDK algorithms are not replaced by the
  release-surface cleanup. The changes above alter execution/analysis semantics,
  diagnostics, failure handling, provenance, portability and packaging where
  explicitly stated.

## 0.0.2 — Beta

Compared with v0.0.1, commit `04344e0376989a63486e206665a6373c48386836`.

### Fixed

- Bind the requested IC Fourier support (`ic.max_mode_per_axis`, including
  omission) into restart and MPI numerical identities. A generated run can no
  longer resume with a different support while labelling its output with the
  replacement value.
- Bound MPI snapshot verification buffers by the largest actual rank payload and
  the configured transfer limit.
- Synchronize rank-local runtime setup errors before any MPI rank returns from
  initialization or starts cleanup.
- Synchronize validation-report publication errors before advancing to memory
  diagnostics.
- Preserve coordinate-reflection symmetry in T-web classification on even meshes;
  mixed tidal derivatives vanish when either differentiated axis is at Nyquist.

### Maintenance

- Share the reflection-compatible Fourier Hessian between T-web and LPT.
- Add opt-in native numerical/runtime regression tests for the 0.0.2 release.
- Document MPI launch agreement and collective force-callback/cache contracts.
- Advance build and CLI version to 0.0.2 while retaining Beta status.

### Compatibility

0.0.2 generated checkpoints used numbered beta restart identities and retained
legacy compatibility rules specific to that release. Those beta compatibility
contracts are not carried forward into the canonical 0.0.3 native format.

## 0.0.1 — Beta

Initial public HYOWON release.
