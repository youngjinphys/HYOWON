# Changelog

## 0.0.2 — Unreleased (Beta)

Compared with [v0.0.1](https://github.com/youngjinphys/HYOWON/releases/tag/v0.0.1),
commit `04344e0376989a63486e206665a6373c48386836`.

### Fixed

- Bind the requested IC Fourier support (`ic.max_mode_per_axis`, including
  omission) into restart and MPI numerical identities. A generated run can no
  longer resume with a different support while labelling its output with the
  replacement value.
- Bound MPI snapshot verification buffers by the largest actual rank payload
  and the configured transfer limit. A large `snapshot_batch_particles` upper
  bound no longer allocates gigabytes to validate a small snapshot.
- Synchronize rank-local runtime setup errors before any MPI rank returns from
  initialization or starts cleanup. A thread request exceeding one rank's OpenMP
  limit now fails across all ranks instead of allowing asymmetric startup or a
  hang. Preserve automatic per-rank thread selection, the originating exception,
  caller OpenMP settings, and ownership of an externally initialized MPI runtime.
- Synchronize validation-report publication errors before advancing to memory
  diagnostics. If rank zero cannot write `runtime_diagnostics.json`, every rank
  now exits the stage with an error; the final snapshot already written remains
  available. Disabled diagnostics retain their existing execution path.
- Preserve coordinate-reflection symmetry in T-web classification on even
  meshes. Mixed tidal derivatives vanish when either differentiated axis is at
  Nyquist, including when both are. Diagonal derivatives and all non-Nyquist
  multipliers are preserved. Affected grid-scale fields can produce different
  web classifications and filament directions than v0.0.1.

### Maintenance

- Share the reflection-compatible Fourier Hessian between T-web and LPT,
  removing the duplicated Nyquist rule. Supported generated ICs exclude these
  Nyquist modes; their LPT results are unchanged by this correction.
- Add opt-in native regression tests with `HYOWON_BUILD_TESTS=ON`; retain their
  sources in release archives without installing test binaries.
- Cover growth against an independent integral solution, EdS drift/kick factors,
  KDK second-order convergence, PM Fourier amplitudes, TreePM force/potential
  consistency, brute-force periodic neighbors/FoF, and spectral normalization.
- Document MPI launch agreement and collective force-callback/cache contracts.
- Advance the build, CLI and source-package version to 0.0.2; retain Beta status.

### Compatibility and validation scope

New checkpoints use `hyowon.restart.v3` to identify the added IC-support binding.
Legacy v1/v2 checkpoints from **generated ICs are rejected** because they contain
no recoverable original support; continue those checkpoints with v0.0.1.
Legacy **snapshot-based** checkpoints remain readable through their bound
source-snapshot identity. Snapshot schema and public configuration keys are unchanged.

MPI fixes address recoverable local exceptions after MPI startup; they do not provide
recovery from failed MPI processes or rank-inconsistent pre-startup configuration.
Small regression runs do not establish large-volume cosmological convergence or
multi-node portability.

## 0.0.1 — Beta

Initial release in the HYOWON repository.
