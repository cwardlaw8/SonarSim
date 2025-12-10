"""
demo_generate.py - Generate full-order and ROM models for demo

Run this script once to generate and save all demo data.
"""

import os
import numpy as np
import time
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from sklearn.decomposition import TruncatedSVD

from setup_sonar_model import setup_sonar_model
from eval_u_Sonar import (
    eval_u_Sonar_20_const,
    eval_u_20_pulse_hann,
    eval_u_5_const,
)


# =============================================================================
# WAVEFORM DEFINITIONS
# =============================================================================

WAVEFORMS = {
    'cont': eval_u_Sonar_20_const,
    'pulse': eval_u_20_pulse_hann,
    '5hz': eval_u_5_const,
}

TRAINING_WAVEFORM = 'cont'  # ROM is trained on this


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'Nx': 201,
    'Nz': 51,
    'Lx': 2000,
    'Lz': 500,
    'f0': 20,
    'source_position': 'center',
    'hydrophone_config': 'horizontal',
    'dt_factor': 0.5,  # dt = max_dt_FE * dt_factor
    'svd_skip': 5,     # Subsample snapshots for SVD
    'svd_k': 100,      # Number of modes to compute
    't_max': 0.8,      # Maximum simulation time (800 ms)
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_full_order(A, B, C, x_start, eval_u_func, t_sim, dt, dx, dz, n, n_phones, save_snapshots=True):
    """
    Run full-order trapezoidal simulation.
    
    Returns
    -------
    t : array
        Time vector
    y : array
        Hydrophone outputs (n_steps x n_phones)
    X : array or None
        State snapshots (n x n_steps) if save_snapshots=True
    elapsed : float
        Computation time in seconds
    """
    def eval_u_scaled(t):
        return dx * dz * eval_u_func(t)
    
    num_steps = int(np.ceil(t_sim / dt))
    
    # Precompute LU factorization
    I_sparse = eye(n, format='csr')
    LHS = I_sparse - (dt / 2) * A
    RHS_mat = I_sparse + (dt / 2) * A
    LU = splu(LHS.tocsc())
    
    # Allocate arrays
    t = np.zeros(num_steps + 1)
    y = np.zeros((num_steps + 1, n_phones))
    if save_snapshots:
        X = np.zeros((n, num_steps + 1))
    
    x_curr = x_start.flatten().copy()
    B_dense = B.toarray().flatten()
    
    t[0] = 0.0
    y[0] = C @ x_curr
    if save_snapshots:
        X[:, 0] = x_curr
    
    t0 = time.perf_counter()
    for i in range(1, num_steps + 1):
        t_prev = t[i - 1]
        t_curr = t_prev + dt
        
        u_prev = eval_u_scaled(t_prev)
        u_curr = eval_u_scaled(t_curr)
        
        rhs = RHS_mat @ x_curr + (dt / 2) * B_dense * (u_prev + u_curr)
        x_curr = LU.solve(rhs)
        
        t[i] = t_curr
        y[i] = C @ x_curr
        if save_snapshots:
            X[:, i] = x_curr
    
    elapsed = time.perf_counter() - t0
    
    if save_snapshots:
        return t, y, X, elapsed
    return t, y, elapsed


def compute_svd(X, skip=5, k=100):
    """
    Compute truncated SVD on subsampled snapshots.
    
    Returns
    -------
    U : array
        Left singular vectors (n x k)
    S : array
        Singular values (k,)
    """
    X_sub = X[:, ::skip]
    k = min(k, X_sub.shape[1] - 1)
    
    svd = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=5, random_state=42)
    svd.fit(X_sub.T)
    
    U = svd.components_.T
    S = svd.singular_values_
    
    return U, S


def find_stable_qs(A, U, max_q=None):
    """
    Find all stable q values (max Re(eigenvalue) <= 0).
    
    Returns
    -------
    stable_qs : list
        List of stable q values
    """
    if max_q is None:
        max_q = U.shape[1]
    
    stable_qs = []
    for q in range(1, max_q + 1):
        Phi_q = U[:, :q]
        A_pod_q = Phi_q.T @ (A @ Phi_q)
        max_re = np.real(np.linalg.eigvals(A_pod_q)).max()
        if max_re <= 0:
            stable_qs.append(q)
    
    return stable_qs


def build_rom(A, B, C, x_start, U, q):
    """
    Build ROM matrices for given q.
    
    Returns
    -------
    dict with Phi, A_pod, B_pod, C_pod, x0_pod
    """
    Phi = U[:, :q]
    A_pod = Phi.T @ (A @ Phi)
    B_pod = Phi.T @ B.toarray()
    C_pod = C @ Phi
    x0_pod = Phi.T @ x_start
    
    return {
        'Phi': Phi,
        'A_pod': A_pod,
        'B_pod': B_pod,
        'C_pod': C_pod,
        'x0_pod': x0_pod,
        'q': q,
    }


# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_demo_data(output_path='demo_outputs/demo_model.npz'):
    """
    Generate all demo data and save to file.
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 70)
    print("DEMO DATA GENERATION")
    print("=" * 70)
    
    # Track total training time
    total_training_start = time.perf_counter()
    
    # -------------------------------------------------------------------------
    # Setup model
    # -------------------------------------------------------------------------
    print("\n[1/5] Setting up model...")
    setup_start = time.perf_counter()
    
    model = setup_sonar_model(
        Nx=CONFIG['Nx'],
        Nz=CONFIG['Nz'],
        Lx=CONFIG['Lx'],
        Lz=CONFIG['Lz'],
        f0=CONFIG['f0'],
        source_position=CONFIG['source_position'],
        hydrophone_config=CONFIG['hydrophone_config'],
        eval_u=WAVEFORMS[TRAINING_WAVEFORM],
    )
    
    # Extract parameters
    p = model['p']
    A = p['A']
    B = p['B']
    x_start = model['x_start']
    t_sim = min(model['t_stop'], CONFIG['t_max'])  # Limit to t_max
    max_dt_FE = model['max_dt_FE']
    dt = max_dt_FE * CONFIG['dt_factor']
    
    Nx, Nz = model['Nx'], model['Nz']
    Lx, Lz = model['Lx'], model['Lz']
    dx, dz = model['dx'], model['dz']
    N = Nx * Nz
    n = 2 * N
    
    # Build C matrix
    hydro = model['hydrophones']
    n_phones = hydro['n_phones']
    
    if 'z_pos' in hydro:
        z_idx = hydro['z_pos']
        x_indices = hydro['x_indices']
        C = np.zeros((n_phones, n))
        for i, x_idx in enumerate(x_indices):
            obs_idx = N + x_idx * Nz + z_idx
            C[i, obs_idx] = 1.0
    else:
        x_indices = hydro['x_indices']
        z_indices = hydro['z_indices']
        C = np.zeros((n_phones, n))
        for i, (x_idx, z_idx) in enumerate(zip(x_indices, z_indices)):
            obs_idx = N + x_idx * Nz + z_idx
            C[i, obs_idx] = 1.0
    
    setup_time = time.perf_counter() - setup_start
    
    print(f"  Grid: {Nx} x {Nz} = {N:,} cells")
    print(f"  DOFs: {n:,}")
    print(f"  dt: {dt*1e6:.2f} µs")
    print(f"  t_sim: {t_sim*1000:.0f} ms")
    print(f"  Setup time: {setup_time:.2f}s")
    
    # -------------------------------------------------------------------------
    # Run full-order for training waveform (with snapshots for SVD)
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Running full-order for training waveform ('{TRAINING_WAVEFORM}')...")
    
    t_train, y_train, X_train, time_train = run_full_order(
        A, B, C, x_start, WAVEFORMS[TRAINING_WAVEFORM],
        t_sim, dt, dx, dz, n, n_phones, save_snapshots=True
    )
    print(f"  Simulation time: {time_train:.2f}s")
    print(f"  Snapshots: {X_train.shape}")
    
    # -------------------------------------------------------------------------
    # Compute SVD and find stable q
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Computing SVD and finding stable modes...")
    
    svd_start = time.perf_counter()
    U, S = compute_svd(X_train, skip=CONFIG['svd_skip'], k=CONFIG['svd_k'])
    svd_time = time.perf_counter() - svd_start
    
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    print(f"  SVD time: {svd_time:.2f}s")
    print(f"  Modes: {U.shape[1]}")
    print(f"  Energy in 10 modes: {cumulative_energy[9]*100:.1f}%")
    print(f"  Energy in 50 modes: {cumulative_energy[min(49, len(S)-1)]*100:.1f}%")
    
    stability_start = time.perf_counter()
    stable_qs = find_stable_qs(A, U)
    stability_time = time.perf_counter() - stability_start
    
    q_pod = max(stable_qs)
    print(f"  Stability check time: {stability_time:.2f}s")
    print(f"  Stable q values: {len(stable_qs)} found")
    print(f"  Max stable q: {q_pod}")
    print(f"  Energy at q={q_pod}: {cumulative_energy[q_pod-1]*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Build ROM
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Building ROM (q={q_pod})...")
    
    build_start = time.perf_counter()
    rom = build_rom(A, B, C, x_start, U, q_pod)
    build_time = time.perf_counter() - build_start
    
    print(f"  A_pod: {rom['A_pod'].shape}")
    print(f"  Compression: {100 * q_pod / n:.4f}%")
    print(f"  Build time: {build_time:.2f}s")
    
    # Total training time (setup + snapshot generation + SVD + stability + build)
    total_training_time = setup_time + time_train + svd_time + stability_time + build_time
    
    # -------------------------------------------------------------------------
    # Run full-order for all waveforms (WITH snapshots for spatial error)
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Running full-order for all waveforms (with snapshots)...")
    
    full_order_results = {}
    
    for name, u_func in WAVEFORMS.items():
        print(f"  {name}...", end=" ")
        
        if name == TRAINING_WAVEFORM:
            # Already computed
            full_order_results[name] = {
                't': t_train,
                'y': y_train,
                'X': X_train,
                'time': time_train,
            }
            print(f"(cached) {time_train:.2f}s")
        else:
            t_fo, y_fo, X_fo, time_fo = run_full_order(
                A, B, C, x_start, u_func,
                t_sim, dt, dx, dz, n, n_phones, save_snapshots=True
            )
            full_order_results[name] = {
                't': t_fo,
                'y': y_fo,
                'X': X_fo,
                'time': time_fo,
            }
            print(f"{time_fo:.2f}s | X: {X_fo.shape}")
    
    total_elapsed = time.perf_counter() - total_training_start
    
    # -------------------------------------------------------------------------
    # Save everything
    # -------------------------------------------------------------------------
    print(f"\nSaving to {output_path}...")
    
    save_data = {
        # Grid parameters
        'Nx': Nx,
        'Nz': Nz,
        'Lx': Lx,
        'Lz': Lz,
        'N': N,
        'n': n,
        'dx': dx,
        'dz': dz,
        
        # Time parameters
        'dt': dt,
        't_sim': t_sim,
        'max_dt_FE': max_dt_FE,
        
        # Hydrophones
        'n_phones': n_phones,
        'phone_idx': n_phones // 2,  # Default to middle hydrophone
        
        # System matrices (sparse stored for reconstruction)
        'A_data': A.data,
        'A_indices': A.indices,
        'A_indptr': A.indptr,
        'A_shape': A.shape,
        'B_data': B.data,
        'B_indices': B.indices,
        'B_indptr': B.indptr,
        'B_shape': B.shape,
        'C': C,
        'x_start': x_start,
        
        # SVD results
        'U': U,
        'S': S,
        'stable_qs': np.array(stable_qs),
        'q_pod': q_pod,
        
        # ROM matrices
        'Phi': rom['Phi'],
        'A_pod': rom['A_pod'],
        'B_pod': rom['B_pod'],
        'C_pod': rom['C_pod'],
        'x0_pod': rom['x0_pod'],
        
        # Waveform names
        'waveform_names': list(WAVEFORMS.keys()),
        'training_waveform': TRAINING_WAVEFORM,
        
        # Timing info
        'training_time': total_training_time,
        'setup_time': setup_time,
        'snapshot_time': time_train,
        'svd_time': svd_time,
        'stability_time': stability_time,
        'build_time': build_time,
    }
    
    # Save full-order results for each waveform (including snapshots)
    for name in WAVEFORMS.keys():
        fo = full_order_results[name]
        save_data[f'{name}_t_fo'] = fo['t']
        save_data[f'{name}_y_fo'] = fo['y']
        save_data[f'{name}_X_fo'] = fo['X']  # Full state snapshots
        save_data[f'{name}_fo_time'] = fo['time']
    
    np.savez_compressed(output_path, **save_data)
    
    # Report file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Saved: {output_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Waveforms: {list(WAVEFORMS.keys())}")
    print(f"Training: '{TRAINING_WAVEFORM}'")
    print(f"ROM: q={q_pod} modes")
    print("-" * 70)
    print("TIMING BREAKDOWN:")
    print(f"  Model setup:        {setup_time:.2f}s")
    print(f"  Snapshot generation:{time_train:.2f}s")
    print(f"  SVD computation:    {svd_time:.2f}s")
    print(f"  Stability check:    {stability_time:.2f}s")
    print(f"  ROM build:          {build_time:.2f}s")
    print(f"  --------------------------------")
    print(f"  TOTAL TRAINING:     {total_training_time:.2f}s")
    print(f"  Total script time:  {total_elapsed:.2f}s")
    print("=" * 70)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    generate_demo_data()