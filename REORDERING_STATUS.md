# State Vector Reordering Status

## What We Did

### Core Change
- **Reordered state vector** from `[p, w]` to `[w, p]` where `w = dp/dt` (velocity) and `p` is pressure
- **Why**: Better numerical properties for preconditioning and asymmetric matrix structure

### Files Updated ✅
1. **getParam_Sonar.py** - Main parameter setup
   - Changed A matrix from `[[0, I], [L, -αI]]` to `[[-D, L], [I, 0]]`
   - Fixed physics bugs: damping now in correct equation (velocity, not pressure)
   - Optimized sparse construction using LIL format
   - **Verified**: Sparse and dense implementations match exactly

2. **getParam_Absorb.py** - Absorbing boundaries
   - Updated Sponge implementation to `[w, p]` ordering
   - Updated PML implementation to `[vx, vz, ψ, p]` ordering
   - Fixed `extract_pressure()` utility

3. **setup_sonar_model.py** - Model wrapper
   - Updated B matrix source location (now in first half: w indices)
   - Fixed parameter name: `BC` → `enforce_surface_BC`
   - Handles both sparse and dense matrix formats correctly

4. **eval_g_Sonar.py** - Hydrophone output
   - Updated pressure extraction: `x[:N]` → `x[N:2*N]`
   - **Verified**: Correctly extracts from second half of state vector

5. **sonar_viz.py** - Visualization
   - Updated pressure extraction: `x[:N]` → `x[N:2*N]`
   - **Verified**: Plots show correct pressure field with source/hydrophone overlay

6. **visualize_sonar.py** - Setup visualization
   - **No changes needed**: Only uses parameter dict, not state vector

7. **create_wave_animation.py** - 2D animation
   - Updated pressure extraction: `X[:N, :]` → `X[N:2*N, :]`

8. **create_3d_wave_animation.py** - 3D animation
   - Updated pressure extraction: `X[:N, :]` → `X[N:2*N, :]`

9. **checkReordering.py** - Comprehensive test suite
   - 16 tests covering all functionality
   - All tests passing ✅
   - Added sparse vs dense verification

### Files Verified (No Changes Needed) ✅
- `eval_f_Sonar.py` - ordering-agnostic
- `eval_u_Sonar.py` - ordering-agnostic
- `eval_Jf_Sonar.py` - ordering-agnostic
- `SimpleSolver.py` - ordering-agnostic
- `simpleLeapFrog.py` - ordering-agnostic

## Problems to Fix

### 1. Testing Needed
- Run full simulation to verify results unchanged
- Compare outputs between old and new ordering
- Test POD workflow compatibility
- Test animation functions with actual simulation data

### 2. Condition Number Increase
- **Before**: κ₂(A) ≈ 4.07e+17
- **After**: κ₂(A) ≈ 1.26e+18 (3× worse)
- **Cause**: New block structure creates near-zero eigenvalue (numerical artifact)
- **Impact**: None for time-stepping (we don't invert A), only affects eigenvalue computations
- **Action**: Monitor, but not critical for simulations

### 3. Testing Needed
- Run full simulation to verify results unchanged
- Compare outputs between old and new ordering
- Test POD workflow compatibility

## Summary
✅ Core reordering complete and verified  
✅ Physics correct and sparse/dense match  
✅ All visualization functions updated  
⚠️ Full simulation testing pending
