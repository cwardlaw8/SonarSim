## Near-Term Tasks

- [ ] **Matrix layout change**: finish swapping p/w ordering in `getParam_Sonar.py`, update any dependent shapes, and re-run `test_Sonar.py` to confirm Jacobian/condition number checks still match expectations.
- [ ] **Baseline runs**: run high-resolution simulations via `setup_sonar_model.py` + `LeapfrogSolver` (and Trap if available) with CFL-safe `dt`; store outputs and timings for the report performance table.
- [ ] **Boundary comparison**: benchmark default absorbing vs. sponge/PML (`getParam_Absorb.py`) for reflection suppression and visual quality; pick one for final figures.
- [ ] **MOR integration**: plug Camille’s reduction code into the state-space model (A, B, C) and generate reduced-order hydrophone traces; quantify errors (L2, peak) and speedups against baseline.
- [ ] **Stability/accuracy sweeps**: use the Δt sweep utilities to map stability boundaries and convergence; capture plots for the report.
- [ ] **Validation plots**: produce spherical spreading check (1/r) and any other analytic comparisons; replace placeholders in `report/main.pdf`.
- [ ] **Animations**: generate clean 2D and 3D GIFs (full-order and MOR overlay/residual) using `create_wave_animation.py` / `create_3d_wave_animation.py`; include hydrophone markers.
- [ ] **Hydrophone outputs**: plot per-hydrophone time series and spectra for key scenarios to support discussion of propagation and attenuation.
- [ ] **Presentation/report polish**: fill in performance table, stability figure, and ethics/limitations notes; align with Wednesday deadline.
- [ ] **Reproducibility**: add a small driver script/notebook to run a standard scenario end-to-end (model setup → solver → plots/animations) with seeded randomness and logged parameters.
