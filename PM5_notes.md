# 20251123 - Tasks for task H

- [-] Run a small-grid damping sweep (vary $\alpha$) and check eigenvalues / stability regions using the PM5 Task C setup.
- [-] Choose a “golden” configuration (grid + $\alpha$) shared between Task C and Task H for all MOR comparisons.
- [-] Add stability metrics to PM5_TaskH (eigenvalues of $\hat{A}$, max $\operatorname{Re}(\lambda)$, and basic energy/norm plots for full vs reduced).
- [ ] Implement and test an eigenmode-truncation reduced model (mode selection guided by frequency band and B/C weights).
- [-] Try a small multi-frequency snapshot design on the small grid (to echo Manny’s band snapshot idea without 22 GB matrices).
- [-] Add a discussion section in PM5_TaskH explaining why plain POD+Galerkin is not enough and motivating structure-preserving MOR.

# 20251124 - Stability sweep outcome
- Small-grid sweep (Nx=60, Nz=30, Lx=7.375, Lz=3.625) across $\alpha$ ∈ {1e-4, 1e-2, 0.5, 1.0} shows max $\operatorname{Re}(\lambda)$ < 0 for all cases; $dt_{\max,\mathrm{FE}}$ ≈ 2.95e-5 (set by CFL).
- $\alpha$ = 1.0 is acceptable and will be our “golden” damping for Task C/H comparisons; keep the same grid/$\alpha$ in the notebook for all MOR baselines.
- Use this $\alpha$=1.0 damping override on the pressure-velocity block of A; pair with Leapfrog $dt \approx 0.5\, dt_{\max,\mathrm{FE}}$ or implicit (Radau/Trapezoidal) for larger grids.

# 20251125 - Notebook run (golden grid $\alpha$=1.0)
- Reference: 60×30 grid, $dt_{\max,\mathrm{FE}}$≈3.99e-4, confidence error ~1.7e-4.
- Baseline POD: $q=60$ hits hydro_err≈1.75e-4/state_err≈1.7e-4 but reduced spectrum has positive max $\operatorname{Re}(\hat{\lambda})$ (~8.9e1) → needs stabilization.
- Output-scaled POD: best at $q=60$ hydro_err≈2.4e-3; still above target and unstable spectrum.
- Weighted augmented POD: with milder weights now hydro_err≈2.8e-3 ($q=60$) but spectrum still unstable (max $\operatorname{Re}(\hat{\lambda})$ ~4.0e2); needs further tuning.
- Multi-frequency POD (band snapshots): $q=60$ hydro_err≈1.7e-1; unstable spectrum (max $\operatorname{Re}(\hat{\lambda})$ ~1.25e2).
- Eigenmode truncation (shift-invert near 3 kHz): overflow/inf errors; unusable as-is.
- Krylov ($s=0$): hydro_err ≈30+ with positive max $\operatorname{Re}(\hat{\lambda})$; needs shift and stabilization.
- Stability table: all reduced bases currently have max $\operatorname{Re}(\hat{\lambda})$ > 0 even when errors are small; next step is a structure-/stability-preserving projection (energy inner product or damping tweak) plus retuned scaling/weights. Full system is stable (max $\operatorname{Re}(\lambda)$ ≈ -0.5).
- W-inner-product POD sweep (pressure=1,2; velocity=5,10,15,20): best is wp=2, wv=10, $q=60$ with hydro_err≈1.8e-4 and max $\operatorname{Re}(\hat{\lambda})$ ~1.3e1 (still >0). wv>10 destabilizes; lower q inaccurate. No reduced basis with max $\operatorname{Re}(\hat{\lambda})$<0 yet.
- Full system stability check ($\alpha$=1.0, 60×30, Lx=Lz=100): dense eig shows max $\operatorname{Re}(\lambda)$ ≈ -0.5, min $\operatorname{Re}(\lambda)$ ≈ -0.5 → full system is stable; increasing $\alpha$ would only add damping and won’t fix reduced-model instability by itself.

# 20251125 – High-level overview and POD stability

## Project overview (Task H context)

- We model a linear sonar propagation system: a damped wave equation discretized on a 60×30 grid over 100 m × 100 m, cast as $\frac{dx}{dt} = A x + B u$, $y = C x$ (outputs at 5 hydrophones).
- Golden configuration:
  - Grid: Nx=60, Nz=30, Lx=Lz=100.
  - Global damping: $\alpha$ = 1.0 applied on the velocity block; boundary absorption ≈5.
  - Full system stability: dense eig of `A` gives max $\operatorname{Re}(\lambda)$ ≈ −0.5, min $\operatorname{Re}(\lambda)$ ≈ −0.5 → full model is stable.
- Reference trajectory:
  - Input: band-limited ping around 3 kHz (scaled by 1e6).
  - Integrator: RK45 with tight tolerances; confidence error between nominal and tighter runs ≈1.7e−4.
  - This confidence level is our target accuracy for reduced models.

## Summary of MOR attempts and results

- Baseline POD (energy-only, single-ping snapshots):
  - $q=60$ achieves hydrophone error near the confidence level (hydro_err ≈1.75e−4).
  - However, the reduced operator `$\hat{A}$ = Vᵀ A V` has max $\operatorname{Re}(\hat{\lambda})$ ≈ +8.9e1 → the reduced system is **unstable**, even though the full system is stable.
- Output-aware / augmented POD:
  - Scaling B and C into the snapshot set improves low-q behavior somewhat but:
    - Best case at $q=60$ still has hydro_err ≈2.4e−3 > target.
    - Reduced spectra show max $\operatorname{Re}(\hat{\lambda})$ ≈ +9.0e1 → unstable.
- Weighted augmented POD:
  - With very aggressive weights, runs blow up (hydro_err up to ~1e52, huge state errors).
  - With milder weights, we avoid overflow but still get hydro_err ≈2.8e−3 at $q=60$ and max $\operatorname{Re}(\hat{\lambda})$ ≈ +4.0e2 → still unstable and not accurate enough.
- Multi-frequency snapshot POD:
  - Using sinusoidal inputs around the ping band improves coverage, but best $q=60$ only gets hydro_err ≈1.7e−1.
  - Reduced spectra show max $\operatorname{Re}(\hat{\lambda})$ ≈ +1.25e2 → unstable.
- Krylov (moment matching, $s=0$):
  - All tested q have hydro_err ≈O(10) or worse; reduced spectra show positive real parts.
  - As expected, $s=0$ moment matching is poor for oscillatory, band-limited inputs.
- Eigenmode truncation (shift-invert near 3 kHz):
  - Our shift-invert eigensolve attempts overflow / return inf errors at the tested sizes.
  - No usable reduced basis was obtained from this approach.

## W-inner-product (stability-weighted) POD

- Motivation: the full system is a damped wave operator with an underlying energy inner product; Euclidean POD does not respect this structure.
- We tried a diagonal surrogate for the energy inner product by weighting pressures and velocities differently:
  - W-POD attempts: pressure weight wp ∈ {1, 2}, velocity weight wv ∈ {5, 10, 15, 20}, q ∈ {40, 60}.
- Results:
  - Several choices preserve good accuracy at $q=60$:
    - E.g., wp=1, wv=5 and wp=2, wv=10 both give hydro_err ≈1.8e−4.
  - These W-POD bases reduce but do not remove instability:
    - For wp=2, wv=10, $q=60$ we get max $\operatorname{Re}(\hat{\lambda})$ ≈ +1.3e1 (better than +8.9e1, but still > 0).
    - Higher velocity weights tend to destabilize badly (very large errors).
  - Lower q (e.g., 40) remains significantly inaccurate.
- Conclusion from W-POD:
  - Diagonal energy weighting and Euclidean QR helps but does **not** produce a stable reduced operator.
  - We have not found any reduced basis with max $\operatorname{Re}(\hat{\lambda})$ < 0 while keeping hydro_err near the confidence level.

## Why POD+Galerkin fails to preserve stability here

- The full system is stable because there exists an energy matrix P such that:
  - `Aᵀ P + P A ≺ 0`, i.e., the energy decays over time.
- Plain POD constructs V that is orthonormal in the **Euclidean** norm (or a crude diagonal surrogate for P) and then uses a Galerkin projection:
  - `$\hat{A}$ = Vᵀ A V`.
- For damped wave / Hamiltonian-like systems, this projection is **not** structure-preserving:
  - It does not maintain the Lyapunov inequality or passivity structure.
  - As a result, even though A is Hurwitz, $\hat{A}$ can (and does) have eigenvalues with $\operatorname{Re}(\hat{\lambda})$ > 0.
- Our experiments match the literature on POD for wave-like systems:
  - Energy-only POD: good short-time accuracy at high q, but unstable reduced dynamics.
  - Output-aware and weighted variants: can improve I/O accuracy but still lose the underlying energy structure, so they inherit instability.
  - W-POD (diagonal energy weights): reduces but does not eliminate unstable eigenvalues.

## Relation to time integrators (Trapezoidal, Leapfrog)

- Changing the ODE integrator (e.g., from RK45 to Trapezoidal or Leapfrog) affects:
  - Step-size stability regions.
  - CFL restrictions and numerical dispersion.
- It does **not** change the eigenvalues of $\hat{A}$:
  - If $\operatorname{Re}(\hat{\lambda})$ > 0, the continuous-time reduced system is unstable.
  - Any reasonable integrator (Forward Euler, Leapfrog, Trapezoidal, Radau, RK45) will reflect that instability; A-stable methods won’t “fix” the model, they just integrate the unstable dynamics more robustly.
- Therefore:
  - For the **full** model, choosing an implicit integrator (e.g., Trapezoidal/Radau) or Leapfrog with a safe dt is very important.
  - For the **reduced** models, changing the integrator cannot solve the fundamental problem that `$\hat{A}$` is not Hurwitz.

## Overall conclusion for Task H

- Full damped sonar model is stable and well-behaved at $\alpha$=1.0.
- Plain POD+Galerkin (even with output-aware scaling, multi-frequency snapshots, and diagonal energy weights) is **not sufficient** for our sonar system:
  - We can hit the error target only at high reduced orders (q≈60).
  - All such reduced models still have some eigenvalues with $\operatorname{Re}(\hat{\lambda})$ > 0.
- A **structure- and stability-preserving MOR** method (e.g., port-Hamiltonian reduction, symplectic/energy inner-product projections, or second-order-preserving MOR) would be more appropriate, but implementing it is beyond our current scope.
- We therefore present our POD-based results as a careful negative result:
  - They demonstrate the practical limitations of naive POD–Galerkin on this damped wave system.
  - They motivate future work on structure-preserving MOR tailored to sonar and wave propagation problems.


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
    - Global absorption $\alpha$ is currently `0.0001` (very small).
    - Boundary absorption is `5` (large).
    - This combination leads to real eigenvalues (not purely imaginary), which hurts the wave-equation-like stability structure.
  - For MOR, we suggest:
    - Increasing $\alpha$ via `getParam_Sonar` (now accepts an alpha argument) so the system is more strongly damped and has no positive real eigenvalues.
    - On a small grid, computing eigenvalues explicitly and checking that there are no eigenvalues with positive real part for the chosen timestep `dt`.
    - Using the eigenvalue code Manny pushed for Task C as a reference.

- **Integrator / time-step observations (Task C)**
  - The $dt_{\max,\mathrm{FE}}$ returned from `getParam_Sonar` is effectively too optimistic for Leapfrog: in practice, stability requires $dt \approx 0.5\, dt_{\max,\mathrm{FE}}$.
  - With this tighter CFL, Leapfrog is stable but still sensitive; Trapezoidal (`solve_ivp` implicit methods) remains stable at larger `dt` and is attractive when we can tolerate some error on the second hydrophone.
  - For large grids, MOR would be most beneficial because running with truly stable `dt` values is expensive; there is also significant aliasing at higher frequencies that needs to be considered in how we choose grid and time step.

- **Stability sweep findings (small grid)**
  - Scanning $\alpha$ ∈ {1e-4, 1e-2, 0.5, 1.0} on the small grid: max $\operatorname{Re}(\lambda)$ < 0 in all cases (~ -$\alpha$/2). Damping shifts the spectrum left but does not change $dt_{\max,\mathrm{FE}}$ (≈2.95e-5), which is set by the Laplacian CFL.
  - For Leapfrog, still use $dt \approx 0.5\, dt_{\max,\mathrm{FE}}$; damping does not loosen the CFL.
  - For MOR tests, adopt a damped “golden” model (e.g., alpha 0.5–1.0) and prefer implicit (Trapezoidal/Radau) for timing on larger grids; keep dt small enough for 3 kHz to avoid aliasing even if the integrator is stable.

- **Where things are / recommendations**
  - Manny will focus on explicit vs. implicit integrator comparisons and likely not push POD much further in time for PM5.
  - He thinks our results are still presentable if we clearly state:
    - POD, even with various scalings, is not sufficient here because the Galerkin projection does not preserve the stability structure of the system.
    - A stability/structure-preserving MOR (e.g., mode truncation or other methods) would be more appropriate.
  - He recommends:
    - Trying larger $\alpha$ values in our notebooks and re-running the MOR comparisons.
    - Verifying eigenvalues (for a small grid) to ensure no positive real parts for the chosen `dt`.
