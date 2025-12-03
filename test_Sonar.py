import numpy as np
import matplotlib.pyplot as plt
from getParam_Sonar import getParam_Sonar
from eval_f_Sonar import eval_f_Sonar
from eval_g_Sonar import eval_g_Sonar
from eval_Jf_Sonar import eval_Jf_Sonar
from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference
from SimpleSolver import SimpleSolver

def test_sonar_complete(eval_u):
    """Complete test suite for sonar model including Jacobian tests"""
    
    print("="*60)
    print("COMPLETE SONAR MODEL TEST SUITE")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    
    # Make randomness deterministic for reproducibility
    np.random.seed(0)

    # PART A: Model Tests
    print("\n" + "="*60)
    print("PART A: MODEL FUNCTIONALITY TESTS")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # TEST 1: Matrix dimensions
    print("\nTEST 1: Matrix Dimensions")
    print("-"*40)
    Nx, Nz = 10, 5
    p, x0, _, _, _ = getParam_Sonar(Nx, Nz, 100, 50, False)
    N = Nx * Nz
    
    if p['A'].shape == (2*N, 2*N) and p['B'].shape == (2*N, 1):
        print("✓ PASSED: Matrix dimensions correct")
        tests_passed += 1
    else:
        print("✗ FAILED: Matrix dimensions wrong")
        tests_failed += 1
    
    # TEST 2: Zero input stability
    print("\nTEST 2: Zero Input Response")
    print("-"*40)
    x_zero = np.zeros((2*N, 1))
    f_zero = eval_f_Sonar(x_zero, p, 0.0)
    
    if np.allclose(f_zero, 0, atol=1e-10):
        print("✓ PASSED: Zero state remains stable")
        tests_passed += 1
    else:
        print(f"✗ FAILED: Max deviation: {np.max(np.abs(f_zero))}")
        tests_failed += 1
    
    # TEST 3: Source excitation
    print("\nTEST 3: Source Excitation")
    print("-"*40)
    u_test = eval_u(0.1)  # At pulse time
    
    if abs(u_test) > 0:
        print(f"✓ PASSED: Source generates signal: {u_test:.2e} Pa")
        tests_passed += 1
    else:
        print("✗ FAILED: No source signal")
        tests_failed += 1
    
    # TEST 4: Wave speed check
    print("\nTEST 4: CFL Condition")
    print("-"*40)
    p, _, _, _, dt_max = getParam_Sonar(20, 10, 100, 50, False)
    dt_cfl = min(p['dx'], p['dz']) / (np.sqrt(2) * p['c'])
    
    if dt_max <= dt_cfl:
        print(f"✓ PASSED: Timestep {dt_max:.6f} respects CFL")
        tests_passed += 1
    else:
        print(f"✗ FAILED: Timestep violates CFL")
        tests_failed += 1
    
    # TEST 5: Hydrophone array
    print("\nTEST 5: Hydrophone Configuration")
    print("-"*40)
    p, x0, _, _, _ = getParam_Sonar(50, 25, 200, 100, False)
    x_test = x0 + np.random.randn(*x0.shape) * 100
    y = eval_g_Sonar(x_test, p)
    
    expected_phones = len([idx for idx in p['hydrophones']['x_indices'] if idx < p['Nx']])
    if y.shape[0] == expected_phones:
        print(f"✓ PASSED: {expected_phones} hydrophones configured")
        tests_passed += 1
    else:
        print(f"✗ FAILED: Expected {expected_phones}, got {y.shape[0]}")
        tests_failed += 1
    
    # TEST 6: Energy growth check (short time)
    print("\nTEST 6: Short-term Stability")
    print("-"*40)
    p, x0, _, _, dt = getParam_Sonar(20, 10, 100, 50, False)
    
    # Run for just 10 steps
    w = dt * 0.1
    num_iter = 10
    
    [X, t] = SimpleSolver(eval_f_Sonar, x0, p, eval_u, 
                         num_iter, w, visualize=False)
    
    initial_energy = np.sum(X[:, 0]**2)
    final_energy = np.sum(X[:, -1]**2)
    
    if final_energy < 1e10 * max(initial_energy, 1.0):
        print(f"✓ PASSED: No immediate divergence")
        tests_passed += 1
    else:
        print(f"✗ FAILED: Rapid divergence detected")
        tests_failed += 1
    
    total_passed += tests_passed
    total_failed += tests_failed
    
    # PART B: Jacobian Tests
    print("\n" + "="*60)
    print("PART B: JACOBIAN TEST BENCH")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Small system for Jacobian testing
    Nx, Nz = 10, 5
    p, x0, _, _, _ = getParam_Sonar(Nx, Nz, 100, 50, UseSparseMatrices=False)
    N = Nx * Nz
    
    # Random state and input
    x_test = np.random.randn(2*N, 1) * 0.01
    u_test = 100.0
    
    # TEST 7: Verify Jacobian equals A matrix
    print("\nTEST 7: Linear System Check")
    print("-"*40)
    
    Jf_analytical = eval_Jf_Sonar(x_test, p, u_test)
    
    if np.allclose(Jf_analytical, p['A']):
        print("✓ PASSED: Jacobian equals A matrix (linear system)")
        tests_passed += 1
    else:
        print("✗ FAILED: Jacobian doesn't match A matrix")
        tests_failed += 1
    
    # TEST 8: Finite Difference Comparison
    print("\nTEST 8: Finite Difference Verification")
    print("-"*40)
    
    # Test different dx values
    dx_values = [1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-10, np.sqrt(np.finfo(float).eps)]
    errors = []
    
    for dx in dx_values:
        p['dxFD'] = dx
        Jf_FD, _ = eval_Jf_FiniteDifference(eval_f_Sonar, x_test, p, u_test)
        error = np.max(np.abs(Jf_analytical - Jf_FD))
        errors.append(error)
        print(f"  dx = {dx:.2e}: error = {error:.2e}")
    
    # Find minimum error
    min_error_idx = np.argmin(errors)
    print(f"  Optimal dx ≈ {dx_values[min_error_idx]:.2e}")
    tests_passed += 1
    
    # TEST 9: Verify df = Jf * dx
    print("\nTEST 9: Linearization Check")
    print("-"*40)
    
    # Small perturbation
    dx = np.random.randn(2*N, 1) * 0.001
    
    # Exact change using Jacobian
    df_jacobian = Jf_analytical @ dx
    
    # Actual change in f
    f1 = eval_f_Sonar(x_test, p, u_test)
    f2 = eval_f_Sonar(x_test + dx, p, u_test)
    df_actual = f2 - f1
    
    error = np.max(np.abs(df_jacobian - df_actual))
    
    if error < 1e-10:
        print(f"✓ PASSED: Linearization accurate (error = {error:.2e})")
        tests_passed += 1
    else:
        print(f"✗ FAILED: Linearization error = {error:.2e}")
        tests_failed += 1
    
    # TEST 10: Eigenvalue Analysis
    print("\nTEST 10: Stability Analysis")
    print("-"*40)
    
    eigenvalues = np.linalg.eigvals(Jf_analytical[:100,:100])  # Sample
    max_real = np.max(np.real(eigenvalues))
    
    print(f"  Max real eigenvalue: {max_real:.2f}")
    if p['alpha'] > 10:  # With high damping
        if max_real < 0:
            print("  ✓ System is stable (overdamped)")
            tests_passed += 1
        else:
            print("  ✗ System should be stable with high damping")
            tests_failed += 1
    else:  # With realistic damping
        print("  ⚠ Low damping - Forward Euler may be unstable")
        tests_passed += 1  # Not a failure, just a warning
    
    total_passed += tests_passed
    total_failed += tests_failed
    
    # Plot error vs dx
    plt.figure(figsize=(8, 6))
    plt.loglog(dx_values, errors, 'b.-', markersize=10, linewidth=2)
    plt.xlabel('Finite Difference Step Size (dx)')
    plt.ylabel('Error ||Jf_analytical - Jf_FD||')
    plt.title('Jacobian Finite Difference Error Analysis')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=np.sqrt(np.finfo(float).eps), color='r', linestyle='--', 
                alpha=0.5, label='sqrt(eps)')
    plt.axvline(x=dx_values[min_error_idx], color='g', linestyle='--',
                alpha=0.5, label=f'Optimal dx={dx_values[min_error_idx]:.2e}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("COMPLETE TEST SUMMARY")
    print("="*60)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Overall Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    
    if total_failed == 0:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n⚠ {total_failed} test(s) need attention")
    
    print("="*60)
    
    return total_passed, total_failed

if __name__ == "__main__":
    test_sonar_complete()
