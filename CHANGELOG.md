# Changelog

## 0.1.2 — Maintenance release

This release focuses on runtime and input robustness. It introduces no new
cosmological model, gravity method, or native snapshot/restart schema.

- Avoid an MPI collective stall when provenance setup fails on one rank.
- Reject non-regular inputs without waiting for a FIFO writer; close file handles
  correctly if buffer or ownership allocation fails.
- Distinguish FFTW wisdom files by planner configuration while retaining compatible
  existing cache files and their plans.
- Propagate machine-readable output failures instead of returning truncated JSON;
  use locale-independent numeric formatting.
- Reject empty required analysis paths and options rather than silently omitting
  requested products.
- Simplify the README, clarify numerical limitations, and provide a separate user
  guide. Exclude development-only release checks from the public source tree.

The existing pure-PM numerical scheme, IC generation, timestep arithmetic,
force operators, and physical defaults are unchanged. Matched small-scale Linux
PM executions and restarts showed identical numerical data under the same inputs
and execution configuration; these checks do not certify cosmological convergence,
cross-platform identity, or TreePM accuracy.

## 0.1.1 — Maintenance release

Improved CIC geometry reuse, exact reductions, SHA-256 processing, checkpoint
publication and MPI failure handling. Preserved the v0.1.0 scientific model and
native artifact schemas.

[Release notes](https://github.com/youngjinphys/HYOWON/releases/tag/v0.1.1)

## 0.1.0 — Paper 0 research release

First numbered public research source for Paper 0. The software version identifies
an implementation; it does not constitute a universal precision-cosmology claim.

[Release notes](https://github.com/youngjinphys/HYOWON/releases/tag/v0.1.0)
