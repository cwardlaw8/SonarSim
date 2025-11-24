# 20251123 - Tasks for task H

- [ ] Run a small-grid damping sweep (vary `alpha`) and check eigenvalues / stability regions using the PM5 Task C setup.
- [ ] Choose a “golden” configuration (grid + `alpha`) shared between Task C and Task H for all MOR comparisons.
- [ ] Add stability metrics to PM5_TaskH (eigenvalues of `A_hat`, max Re(λ), and basic energy/norm plots for full vs reduced).
- [ ] Implement and test an eigenmode-truncation reduced model (mode selection guided by frequency band and B/C weights).
- [ ] Try a small multi-frequency snapshot design on the small grid (to echo Manny’s band snapshot idea without 22 GB matrices).
- [ ] Add a discussion section in PM5_TaskH explaining why plain POD+Galerkin is not enough and motivating structure-preserving MOR.

# 20251124 - Stability sweep outcome
- Small-grid sweep (Nx=60, Nz=30, Lx=7.375, Lz=3.625) across α ∈ {1e-4, 1e-2, 0.5, 1.0} shows max Re(λ) < 0 for all cases; dt_max_FE ≈ 2.95e-5 (set by CFL).
- α = 1.0 is acceptable and will be our “golden” damping for Task C/H comparisons; keep the same grid/α in the notebook for all MOR baselines.
- Use this α=1.0 damping override on the pressure-velocity block of A; pair with Leapfrog dt ≈ 0.5·dt_max_FE or implicit (Radau/Trapezoidal) for larger grids.

# 20251125 - Notebook run (golden grid α=1.0)
- Reference: 60×30 grid, dt_max_FE≈3.99e-4, confidence error ~1.7e-4.
- Baseline POD: q=60 hits hydro_err≈1.75e-4/state_err≈1.7e-4 but reduced spectrum has positive max Re(λ̂) (~8.9e1) → needs stabilization.
- Output-scaled POD: best at q=60 hydro_err≈2.4e-3; still above target and unstable spectrum.
- Weighted augmented POD: with milder weights now hydro_err≈2.8e-3 (q=60) but spectrum still unstable (max Re(λ̂) ~4.0e2); needs further tuning.
- Multi-frequency POD (band snapshots): q=60 hydro_err≈1.7e-1; unstable spectrum (max Re(λ̂) ~1.25e2).
- Eigenmode truncation (shift-invert near 3 kHz): overflow/inf errors; unusable as-is.
- Krylov (s=0): hydro_err ≈30+ with positive max Re(λ̂); needs shift and stabilization.
- Stability table: all reduced bases currently have max Re(λ̂) > 0 even when errors are small; next step is a structure-/stability-preserving projection (energy inner product or damping tweak) plus retuned scaling/weights.
- W-inner-product POD (pressure=1, velocity=5): q=60 hydro_err≈1.8e-4 with reduced max Re(λ̂) ~2.6e1 (still >0); lower q inaccurate. Needs stronger stabilization to make max Re(λ̂)<0 while keeping error near 1e-4.


# 20251123 - Manny's feedback

- **Initial assessment of our MOR attempt**
  - Current POD-based model order reduction does not achieve much reduction (needs large q for accuracy).
  - For this sonar problem with a frequency-band input we should be:
    - Building the snapshot matrix from several simulations across a frequency band, not just one time-domain pulse.
    - Comparing reduced models against a *large* reference model where speedup is meaningful.
    - Considering eigenvalue/mode truncation rather than pure POD, since we care more about frequency response than global energy; POD tends to drop late-time, low-energy hydrophone content.

- **Large snapshot experiment and practical limits**
  - He tried constructing a very large “reference matrix” across many frequencies (Nyquist-respecting over a band).
  - This led to a matrix on the order of ~22 GB; SciPy’s dense SVD could not handle it on his machine.
  - Conclusion: brute-force, huge SVD-based POD is not practical for this setup on typical hardware.

- **Conclusions about POD + Galerkin on this system**
  - POD alone is not working well for our sonar model.
  - Once we project using the POD basis (Galerkin), the reduced system no longer preserves the stability/physics structure and can blow up or behave unphysically.
  - Regardless of how we scale snapshots, POD+Galerkin may still:
    - Produce very large state values.
    - Fail to respect the original dynamics and stability.

- **Stability / damping observations and suggested fix**
  - While doing ODE stability checks, we noticed:
    - Global absorption `alpha` is currently `0.0001` (very small).
    - Boundary absorption is `5` (large).
    - This combination leads to real eigenvalues (not purely imaginary), which hurts the wave-equation-like stability structure.
  - For MOR, we suggest:
    - Increasing `alpha` via `getParam_Sonar` (now accepts an alpha argument) so the system is more strongly damped and has no positive real eigenvalues.
    - On a small grid, computing eigenvalues explicitly and checking that there are no eigenvalues with positive real part for the chosen timestep `dt`.
    - Using the eigenvalue code Manny pushed for Task C as a reference.

- **Integrator / time-step observations (Task C)**
  - The `max_dt_FE` returned from `getParam_Sonar` is effectively too optimistic for Leapfrog: in practice, stability requires `dt ≈ 0.5 * max_dt_FE`.
  - With this tighter CFL, Leapfrog is stable but still sensitive; Trapezoidal (`solve_ivp` implicit methods) remains stable at larger `dt` and is attractive when we can tolerate some error on the second hydrophone.
  - For large grids, MOR would be most beneficial because running with truly stable `dt` values is expensive; there is also significant aliasing at higher frequencies that needs to be considered in how we choose grid and time step.

- **Stability sweep findings (small grid)**
  - Scanning `alpha` ∈ {1e-4, 1e-2, 0.5, 1.0} on the small grid: max Re(λ) < 0 in all cases (~ -alpha/2). Damping shifts the spectrum left but does not change `dt_max_FE` (≈2.95e-5), which is set by the Laplacian CFL.
  - For Leapfrog, still use `dt ≈ 0.5 * dt_max_FE`; damping does not loosen the CFL.
  - For MOR tests, adopt a damped “golden” model (e.g., alpha 0.5–1.0) and prefer implicit (Trapezoidal/Radau) for timing on larger grids; keep dt small enough for 3 kHz to avoid aliasing even if the integrator is stable.

- **Where things are / recommendations**
  - Manny will focus on explicit vs. implicit integrator comparisons and likely not push POD much further in time for PM5.
  - He thinks our results are still presentable if we clearly state:
    - POD, even with various scalings, is not sufficient here because the Galerkin projection does not preserve the stability structure of the system.
    - A stability/structure-preserving MOR (e.g., mode truncation or other methods) would be more appropriate.
  - He recommends:
    - Trying larger `alpha` values in our notebooks and re-running the MOR comparisons.
    - Verifying eigenvalues (for a small grid) to ensure no positive real parts for the chosen `dt`.
