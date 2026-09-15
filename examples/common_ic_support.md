# Common initial-condition support

Use optional `ic.max_mode_per_axis` when comparing different particle resolutions
with the same projected initial-condition generator:

```toml
[ic]
# Other required generated-IC settings still apply.
mode = "generate"
mesh_per_dimension = 64
max_mode_per_axis = 7
```

`K = max_mode_per_axis` is an **inclusive Cartesian integer-mode bound**: keep
`|m_x|, |m_y|, |m_z| <= K`. It applies to both the first-order density and the
final quadratic 2LPT source. The latter projection is essential because quadratic
products can generate modes outside the original density support. This defines a
projected 2LPT map, not the untruncated continuous second-order solution.

`K` must be positive and satisfy `2*K < particles_per_dimension`. Without this
option the safe particle-lattice support is used: `K = floor((N-1)/2)`. The option
is rejected for snapshot input, whose particles and generator provenance come from
the admitted file.

For a common-map comparison:

- Use the same nonzero IC mesh dimension `M` at every rung. `M` must be divisible
  by every particle dimension; 2LPT still requires `M >= 2*N` for each rung.
- Fix the spectrum bytes, cosmology, spectrum/start redshifts, seed, amplitude and
  phase-pairing rules, and `K`.
- Control FFT/provider/thread settings separately and compare actual planner
  records when claiming execution identity.
- Compare position and momentum/velocity together. The generator samples mesh
  nodes directly; mathematically coincident base coordinates may still differ by
  floating-point rounding for different lattice ratios.

For example, particle dimensions 16 and 32, common `M=64`, and `K=7` satisfy the
input constraints. This is an interface example, not a converged cosmological
setup. Uniform particle mass changes with particle count as required to keep total
mass fixed.

The input spectrum must cover `2*pi/L` through `sqrt(3)*K*2*pi/L`. Its range must
also support the separate sigma8 normalization diagnostic; narrowing the represented
IC field does not redefine sigma8.

The snapshot `/Config/RunMetadataJson` records `ic_max_mode_per_axis` (requested
value, null when omitted) and `ic_effective_max_mode_per_axis` (applied value).
Explicit support also enters the physics fingerprint. Current native snapshots must
be internally consistent with these coordinates; incompatible artifacts fail closed.

Generated Fourier evidence includes only retained modes inside the existing open
particle-Nyquist sphere. Its mode count and support label therefore describe the
intersection of a Cartesian window and a radial aperture. Shells at different
particle resolutions can cover different radial ranges even when the generated map
uses the same `K`.
