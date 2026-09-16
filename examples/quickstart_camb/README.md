# CAMB-backed quick-start spectrum

From the repository root, after building:

```sh
./build/hyowon examples/quickstart_generate.toml
```

This runs 64 particles on an 8^3 IC/PM mesh from z=9 to z=8 and writes two
snapshots plus restart checkpoints below `runs/quickstart_generate`. Choose a new
`output.run_label` when repeating it. These very small coordinates exercise the
software path; they are not a converged cosmological calculation.

`linear_cb_z0.dat` is CAMB 2.0.4 output containing 2048 positive samples over
1e-5 <= k <= 50 h/Mpc for the synchronous-gauge linear cold-plus-baryon power
spectrum. The generating configuration used h=0.7, Omega_m=0.3, Omega_b=0.05,
n_s=1, zero massive-neutrino mass, and a scalar amplitude calibrated to
sigma8(z=0)=0.8. The stored CAMB parameter dump is `camb_parameters.txt`; hashes
and generation-environment metadata are recorded in `provenance.json`.

CAMB includes photons and massless neutrinos at Tcmb=2.7255 K, so the CAMB flat
background has Omega_Lambda=0.6999146412354531. The HYOWON quick-start config
uses its maintained late-time matter-Lambda model with Omega_Lambda=0.7 and
backscales the z=0 table using that declared background. This projection is
explicit and is not independently validated by the quick-start run.

The checked-in table requires no Python or CAMB installation. Reproducing the
exact generator requires CAMB 2.0.4 and the parameters recorded here; the
repository intentionally does not ship the development-time generator or its
local execution environment.
