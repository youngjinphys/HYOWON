# Common initial-condition support

Use optional `ic.max_mode_per_axis` when comparing particle resolutions with the
same projected initial-condition generator:

```toml
[ic]
# Other required generated-IC settings still apply.
mode = "generate"
mesh_per_dimension = 64
max_mode_per_axis = 7
```

`K = max_mode_per_axis` is an **inclusive Cartesian integer-mode bound**:
`|m_x|, |m_y|, |m_z| <= K`. It applies to the first-order density and to the final
quadratic 2LPT source after projection. This defines a projected 2LPT map, not the
untruncated continuous second-order solution.

K must be positive and satisfy `2*K < N`, where N is
`particles_per_dimension`. Without an explicit K, HYOWON uses
`floor((N-1)/2)`. Snapshot input does not accept generated-IC support coordinates.

## Why projected 2LPT requires `M > 3K`

Let `p` and `q` denote one Cartesian component of two retained input modes. The
exact quadratic product contains convolution component `p+q` with
`p,q in [-K,K]`, hence `p+q in [-2K,2K]`. HYOWON retains only output components
`k in [-K,K]` after the source FFT.

On a periodic M-point DFT, a quadratic contribution can alias into retained k only
if

```text
p + q - k = s M
```

for a nonzero integer s. Since

```text
|p + q - k| <= |p| + |q| + |k| <= 3K,
```

the strict condition `M > 3K` excludes every such alias. Equality is not enough.
For example at `M=3K`, choosing 3D input modes
`p=(K,K,0)` and `q=(K,-K,0)` produces `p+q=(2K,0,0)`, which aliases to
`(-K,0,0)`; the 2LPT angular coupling is nonzero for this pair.

This statement is deliberately limited to the **retained projected band**. The
full quadratic support reaches `[-2K,2K]`. Representing that entire band without
identifying the two edge modes at a Nyquist plane requires `M > 4K`. HYOWON does
not need that stronger condition because modes outside K are discarded immediately
after the quadratic source FFT, before any further nonlinear multiplication.

The condition is derived from the discrete map actually implemented here; no
external de-aliasing rule is used as an authority for it.

## Relation between M and N

`M > 3K` is the quadratic de-aliasing condition. The additional requirement
`M % N == 0` has a different origin: the current LPT implementation samples the IC
mesh directly at particle-lattice nodes using the integer ratio `M/N`. It should
not be interpreted as a fundamental Fourier requirement.

With `ic.mesh_per_dimension = 0`:

- 1LPT uses `M=N`;
- projected 2LPT uses `M=N` when `N > 3K`;
- otherwise the current integer-ratio policy uses `M=2N`.

For default near-particle-Nyquist support the historical `2N` mesh therefore
normally remains. A deliberately reduced common K can use N directly when the
retained-band condition is already satisfied.

## Common-map comparisons

For comparisons across particle resolutions:

- Use the same nonzero M at every rung when exact common mesh arithmetic is part of
  the comparison. M must be divisible by every particle N and projected 2LPT must
  satisfy `M > 3K`.
- Fix the spectrum bytes, cosmology, spectrum/start redshifts, seed, amplitude mode,
  phase-pairing rule, and K.
- Control FFT provider, planner, and thread settings separately when claiming
  execution identity.
- Compare both position and momentum/velocity. The generator samples mesh nodes
  directly; mathematically coincident base coordinates can still differ through
  floating-point coordinate construction at different lattice ratios.

For example, particle dimensions 16 and 32 with common `M=64` and `K=7` satisfy
the interface constraints. This is not a statement that those values form a
scientifically converged cosmological calculation.

The input spectrum must cover `2*pi/L` through `sqrt(3)*K*2*pi/L`. Its range must
also support the separate sigma8 normalization diagnostic; narrowing the represented
IC field does not redefine sigma8.

Native snapshot metadata records both the requested
`ic_max_mode_per_axis` (null when omitted) and the applied
`ic_effective_max_mode_per_axis`. The IC mesh and support enter the generated
artifact provenance. Incompatible current-format artifacts fail closed.

Generated Fourier evidence reports retained modes within its documented radial
aperture. Its shell support is therefore not identical to the Cartesian generator
window; comparisons across N must account for the actual reported shell range.
