"""
demo_funcs.py - Functions for POD-ROM live demo
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt


# =============================================================================
# WAVEFORM DEFINITIONS
# =============================================================================

def eval_u_cont(t):
    """Continuous 20 Hz with smooth ramp-on (training signal)"""
    f0 = 20
    t_ramp = 0.2
    A0 = 100
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)


def eval_u_pulse(t):
    """Hann-windowed pulse at 20 Hz"""
    f0 = 20
    t_duration = 0.25
    A0 = 100
    if t < 0 or t > t_duration:
        return 0.0
    envelope = 0.5 * (1 - np.cos(2 * np.pi * t / t_duration))
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)


def eval_u_5hz(t):
    """5 Hz - out of band (lower frequency)"""
    f0 = 5
    t_ramp = 0.5
    A0 = 100
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)


# Waveform dictionary
WAVEFORMS = {
    'cont': eval_u_cont,
    'pulse': eval_u_pulse,
    '5hz': eval_u_5hz,
}

# Maximum simulation time for demo
T_MAX = 0.75  # 800 ms


# =============================================================================
# LOAD DEMO DATA
# =============================================================================

def load_demo_data(path='demo_outputs/demo_model.npz'):
    """
    Load saved demo model.
    
    Parameters
    ----------
    path : str
        Path to saved demo_model.npz
    
    Returns
    -------
    dict with all model data
    """
    data = np.load(path, allow_pickle=True)
    
    # Grid parameters
    Nx, Nz = int(data['Nx']), int(data['Nz'])
    Lx, Lz = float(data['Lx']), float(data['Lz'])
    N, n = int(data['N']), int(data['n'])
    dx, dz = float(data['dx']), float(data['dz'])
    
    # Time parameters
    dt = float(data['dt'])
    t_sim = min(float(data['t_sim']), T_MAX)  # Limit to 800ms
    
    # Hydrophones
    n_phones = int(data['n_phones'])
    phone_idx = int(data['phone_idx'])
    
    # Reconstruct sparse matrices
    A = csr_matrix((data['A_data'], data['A_indices'], data['A_indptr']), shape=tuple(data['A_shape']))
    B = csr_matrix((data['B_data'], data['B_indices'], data['B_indptr']), shape=tuple(data['B_shape']))
    C = data['C']
    x_start = data['x_start']
    
    # SVD results
    U = data['U']
    S = data['S']
    stable_qs = list(data['stable_qs'])
    q_pod = int(data['q_pod'])
    
    # ROM matrices
    Phi = data['Phi']
    A_pod = data['A_pod']
    B_pod = data['B_pod']
    C_pod = data['C_pod']
    x0_pod = data['x0_pod']
    
    # Waveform info
    waveform_names = list(data['waveform_names'])
    training_waveform = str(data['training_waveform'])
    
    # Timing info
    training_time = float(data['training_time']) if 'training_time' in data.files else None
    setup_time = float(data['setup_time']) if 'setup_time' in data.files else None
    snapshot_time = float(data['snapshot_time']) if 'snapshot_time' in data.files else None
    svd_time = float(data['svd_time']) if 'svd_time' in data.files else None
    stability_time = float(data['stability_time']) if 'stability_time' in data.files else None
    build_time = float(data['build_time']) if 'build_time' in data.files else None
    
    # Full-order results (truncate to T_MAX)
    full_order = {}
    for name in waveform_names:
        t_fo = data[f'{name}_t_fo']
        y_fo = data[f'{name}_y_fo']
        
        # Check for X_full if saved
        X_key = f'{name}_X_fo'
        X_fo = data[X_key] if X_key in data.files else None
        
        # Truncate to T_MAX
        max_idx = np.searchsorted(t_fo, T_MAX)
        full_order[name] = {
            't': t_fo[:max_idx],
            'y': y_fo[:max_idx],
            'X': X_fo[:, :max_idx] if X_fo is not None else None,
            'time': float(data[f'{name}_fo_time']),
        }
    
    data.close()
    
    print(f"Loaded demo model from {path}")
    print(f"  Grid: {Nx} x {Nz} ({n:,} DOFs)")
    print(f"  ROM: q={q_pod} modes")
    print(f"  Waveforms: {waveform_names}")
    print(f"  Training: '{training_waveform}'")
    print(f"  t_max: {T_MAX*1000:.0f} ms")
    if training_time is not None:
        print(f"  Training time: {training_time*1000:.0f}ms")
        if all(t is not None for t in [snapshot_time, svd_time, stability_time, build_time]):
            print(f"    - Snapshots:  {snapshot_time*1000:.0f}ms")
            print(f"    - SVD:        {svd_time*1000:.0f}ms")
            print(f"    - Stability:  {stability_time*1000:.0f}ms")
            print(f"    - Build:      {build_time*1000:.0f}ms")
    
    return {
        # Grid
        'Nx': Nx, 'Nz': Nz, 'Lx': Lx, 'Lz': Lz,
        'N': N, 'n': n, 'dx': dx, 'dz': dz,
        
        # Time
        'dt': dt, 't_sim': t_sim,
        
        # Hydrophones
        'n_phones': n_phones, 'phone_idx': phone_idx,
        
        # System matrices
        'A': A, 'B': B, 'C': C, 'x_start': x_start,
        
        # SVD
        'U': U, 'S': S, 'stable_qs': stable_qs, 'q_pod': q_pod,
        
        # ROM
        'Phi': Phi, 'A_pod': A_pod, 'B_pod': B_pod, 'C_pod': C_pod, 'x0_pod': x0_pod,
        
        # Full-order references
        'full_order': full_order,
        'waveform_names': waveform_names,
        'training_waveform': training_waveform,
        
        # Timing info
        'training_time': training_time,
        'setup_time': setup_time,
        'snapshot_time': snapshot_time,
        'svd_time': svd_time,
        'stability_time': stability_time,
        'build_time': build_time,
    }

# =============================================================================
# SIMULATE ROM
# =============================================================================

def simulate_rom(demo, waveform):
    """
    Simulate ROM with given waveform.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    waveform : str or callable
        Waveform name ('cont', 'pulse', '5hz') or custom function
    
    Returns
    -------
    dict with t, y, x, elapsed
    """
    # Get waveform function
    if isinstance(waveform, str):
        if waveform not in WAVEFORMS:
            raise ValueError(f"Unknown waveform '{waveform}'. Options: {list(WAVEFORMS.keys())}")
        eval_u_func = WAVEFORMS[waveform]
    else:
        eval_u_func = waveform
    
    A_pod = demo['A_pod']
    B_pod = demo['B_pod']
    C_pod = demo['C_pod']
    x0_pod = demo['x0_pod']
    q_pod = demo['q_pod']
    dx, dz = demo['dx'], demo['dz']
    dt = demo['dt']
    t_sim = min(demo['t_sim'], T_MAX)
    n_phones = demo['n_phones']
    
    def eval_u_scaled(t):
        return dx * dz * eval_u_func(t)
    
    t = np.arange(0, t_sim + dt, dt)
    n_steps = len(t)
    
    x = np.zeros((n_steps, q_pod))
    y = np.zeros((n_steps, n_phones))
    
    x_curr = x0_pod.flatten().copy()
    B_flat = B_pod.flatten()
    
    # Trapezoidal method
    I = np.eye(q_pod)
    LHS = I - (dt / 2) * A_pod
    RHS_mat = I + (dt / 2) * A_pod
    LU = lu_factor(LHS)
    
    x[0] = x_curr
    y[0] = C_pod @ x_curr
    
    import time
    t0 = time.perf_counter()
    
    for i in range(1, n_steps):
        u_prev = eval_u_scaled(t[i - 1])
        u_curr = eval_u_scaled(t[i])
        
        rhs = RHS_mat @ x_curr + (dt / 2) * B_flat * (u_prev + u_curr)
        x_curr = lu_solve(LU, rhs)
        
        x[i] = x_curr
        y[i] = C_pod @ x_curr
    
    elapsed = time.perf_counter() - t0
    
    return {
        't': t,
        'y': y,
        'x': x,
        'elapsed': elapsed,
        'waveform': waveform if isinstance(waveform, str) else 'custom',
    }


# =============================================================================
# COMPUTE ERROR
# =============================================================================

def compute_error(demo, rom_result, phone_idx=None):
    """
    Compute error between ROM and full-order.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    rom_result : dict
        Output from simulate_rom()
    phone_idx : int, optional
        Hydrophone index (default: middle)
    
    Returns
    -------
    dict with rel_error, abs_error, etc.
    """
    if phone_idx is None:
        phone_idx = demo['phone_idx']
    
    waveform_name = rom_result['waveform']
    
    if waveform_name not in demo['full_order']:
        raise ValueError(f"No full-order reference for '{waveform_name}'. "
                        f"Available: {list(demo['full_order'].keys())}")
    
    fo = demo['full_order'][waveform_name]
    t_fo, y_fo = fo['t'], fo['y']
    t_rom, y_rom = rom_result['t'], rom_result['y']
    
    # Interpolate ROM to full-order time grid
    y_rom_interp = np.interp(t_fo, t_rom, y_rom[:, phone_idx])
    
    # Relative L2 error
    rel_error = np.linalg.norm(y_fo[:, phone_idx] - y_rom_interp) / \
                (np.linalg.norm(y_fo[:, phone_idx]) + 1e-12) * 100
    
    # Absolute error over time
    abs_error = np.abs(y_fo[:, phone_idx] - y_rom_interp)
    
    return {
        'rel_error': rel_error,
        'abs_error': abs_error,
        't': t_fo,
        'y_fo': y_fo[:, phone_idx],
        'y_rom': y_rom_interp,
    }


# =============================================================================
# RECONSTRUCT PRESSURE
# =============================================================================

def reconstruct_pressure(demo, rom_result, frame_idx):
    """
    Reconstruct pressure field from ROM modal coordinates.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    rom_result : dict
        Output from simulate_rom()
    frame_idx : int
        Time frame index
    
    Returns
    -------
    p : array
        Pressure field (Nx x Nz)
    t : float
        Time in seconds
    """
    Phi = demo['Phi']
    N = demo['N']
    Nx, Nz = demo['Nx'], demo['Nz']
    
    x_modal = rom_result['x'][frame_idx, :]
    x_full = Phi @ x_modal
    
    # Pressure is in second half of state vector
    p = x_full[N:].reshape(Nx, Nz)
    t = rom_result['t'][frame_idx]
    
    return p, t


# =============================================================================
# VISUALIZE INPUT SIGNAL
# =============================================================================

def plot_input_signal(waveform, t_max=T_MAX, n_points=1000):
    """
    Plot input signal and its frequency content.
    
    Parameters
    ----------
    waveform : str or callable
        Waveform name ('cont', 'pulse', '5hz') or custom function
    t_max : float
        Maximum time in seconds (default: 0.8)
    n_points : int
        Number of points for plotting
    """
    # Get waveform function
    if isinstance(waveform, str):
        if waveform not in WAVEFORMS:
            raise ValueError(f"Unknown waveform '{waveform}'. Options: {list(WAVEFORMS.keys())}")
        eval_u_func = WAVEFORMS[waveform]
        title = waveform
    else:
        eval_u_func = waveform
        title = 'Custom'
    
    # Generate time signal
    t = np.linspace(0, t_max, n_points)
    u = np.array([eval_u_func(ti) for ti in t])
    
    # Compute FFT
    dt = t[1] - t[0]
    n_fft = len(u)
    freqs = np.fft.rfftfreq(n_fft, dt)
    fft_vals = np.abs(np.fft.rfft(u)) / n_fft * 2
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Time domain
    axes[0].plot(t * 1000, u, 'b-', lw=1.5)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f"'{title}' - Time Domain")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, t_max * 1000])
    
    # Frequency domain
    axes[1].plot(freqs, fft_vals, 'r-', lw=1.5)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title(f"'{title}' - Frequency Content")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 50])
    
    # Mark 20 Hz (training frequency)
    axes[1].axvline(20, color='green', linestyle='--', alpha=0.7, label='Training (20 Hz)')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_all_inputs(waveforms=None, t_max=T_MAX, n_points=1000):
    """
    Plot specified waveforms and their frequency content.
    
    Parameters
    ----------
    waveforms : list, optional
        List of waveform names to plot (default: all)
    t_max : float
        Maximum time in seconds
    n_points : int
        Number of points for plotting
    """
    if waveforms is None:
        waveforms = list(WAVEFORMS.keys())
    
    # Validate
    for w in waveforms:
        if w not in WAVEFORMS:
            raise ValueError(f"Unknown waveform '{w}'. Options: {list(WAVEFORMS.keys())}")
    
    n_waveforms = len(waveforms)
    
    fig, axes = plt.subplots(n_waveforms, 2, figsize=(12, 3 * n_waveforms))
    
    # Handle single waveform case
    if n_waveforms == 1:
        axes = axes.reshape(1, 2)
    
    t = np.linspace(0, t_max, n_points)
    dt = t[1] - t[0]
    freqs = np.fft.rfftfreq(n_points, dt)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_waveforms))
    
    for i, name in enumerate(waveforms):
        eval_u_func = WAVEFORMS[name]
        u = np.array([eval_u_func(ti) for ti in t])
        fft_vals = np.abs(np.fft.rfft(u)) / n_points * 2
        
        # Time domain
        axes[i, 0].plot(t * 1000, u, '-', color=colors[i], lw=1.5)
        axes[i, 0].set_ylabel(f"'{name}'")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim([0, t_max * 1000])
        if i == 0:
            axes[i, 0].set_title('Time Domain')
        if i == n_waveforms - 1:
            axes[i, 0].set_xlabel('Time (ms)')
        
        # Frequency domain
        axes[i, 1].plot(freqs, fft_vals, '-', color=colors[i], lw=1.5)
        axes[i, 1].axvline(20, color='green', linestyle='--', alpha=0.5)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim([0, 50])
        if i == 0:
            axes[i, 1].set_title('Frequency Content (green = 20 Hz training)')
        if i == n_waveforms - 1:
            axes[i, 1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.show()


def plot_stability_and_energy(demo):
    """
    Plot ROM stability analysis and energy captured per mode.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    """
    U = demo['U']
    S = demo['S']
    A = demo['A']
    stable_qs = demo['stable_qs']
    q_pod = demo['q_pod']
    
    # Compute stability for all q values
    max_q = U.shape[1]
    qs = list(range(1, max_q + 1))
    max_res = []
    
    for q in qs:
        Phi_q = U[:, :q]
        A_pod_q = Phi_q.T @ (A @ Phi_q)
        max_re = np.real(np.linalg.eigvals(A_pod_q)).max()
        max_res.append(max_re)
    
    stable = [q in stable_qs for q in qs]
    
    # Compute cumulative energy
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2) * 100
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Max Re(λ) vs q
    colors = ['green' if s else 'red' for s in stable]
    axes[0].scatter(qs, max_res, c=colors, s=30, alpha=0.7)
    axes[0].axhline(0, color='k', linestyle='--', lw=1, label='Stability boundary')
    axes[0].axvline(q_pod, color='blue', linestyle=':', lw=2, label=f'Selected q={q_pod}')
    axes[0].set_xlabel('Number of modes (q)')
    axes[0].set_ylabel('Max Re(λ)')
    axes[0].set_title('ROM Stability (green=stable, red=unstable)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim([0, max_q + 1])
    
    # Right: Cumulative energy vs q
    axes[1].plot(range(1, len(S) + 1), cumulative_energy, 'b-', lw=2)
    axes[1].axhline(99, color='orange', linestyle='--', lw=1, label='99% energy')
    axes[1].axvline(q_pod, color='blue', linestyle=':', lw=2, label=f'Selected q={q_pod}')
    axes[1].axhline(cumulative_energy[q_pod - 1], color='green', linestyle='--', lw=1, 
                    label=f'{cumulative_energy[q_pod - 1]:.1f}% at q={q_pod}')
    axes[1].set_xlabel('Number of modes (q)')
    axes[1].set_ylabel('Cumulative Energy (%)')
    axes[1].set_title('Energy Captured by POD Modes')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim([0, max_q + 1])
    axes[1].set_ylim([90, 100.5])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    n_stable = len(stable_qs)
    print(f"Stable modes: {n_stable}/{max_q} ({100*n_stable/max_q:.0f}%)")
    print(f"Selected q: {q_pod}")
    print(f"Energy at q={q_pod}: {cumulative_energy[q_pod - 1]:.2f}%")

# =============================================================================
# ANIMATE COMPARISON (Full-order, ROM, Error)
# =============================================================================

def animate_comparison(demo, waveform, n_frames=100, interval=50):
    """
    Animate full-order, ROM, and spatial error as 3 subplots.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    waveform : str
        Waveform name ('cont', 'pulse', '5hz')
    n_frames : int
        Number of animation frames
    interval : int
        Animation interval in ms
    """
    from matplotlib.widgets import Slider, Button
    
    # Validate waveform
    if waveform not in WAVEFORMS:
        raise ValueError(f"Unknown waveform '{waveform}'. Options: {list(WAVEFORMS.keys())}")
    
    if waveform not in demo['full_order']:
        raise ValueError(f"No full-order reference for '{waveform}'")
    
    # Check if full-order snapshots are available
    fo = demo['full_order'][waveform]
    if fo['X'] is None:
        raise ValueError(f"No full-order snapshots (X_fo) saved for '{waveform}'. "
                        "Re-run demo_generate.py with save_snapshots=True.")
    
    # Simulate ROM
    print(f"Simulating ROM for '{waveform}'...")
    rom_result = simulate_rom(demo, waveform)
    print(f"  ROM time: {rom_result['elapsed']*1000:.1f} ms")
    
    # Compute hydrophone error for reference
    err = compute_error(demo, rom_result)
    print(f"  Hydrophone error: {err['rel_error']:.2f}%")
    
    # Extract parameters
    Phi = demo['Phi']
    N, Nx, Nz = demo['N'], demo['Nx'], demo['Nz']
    Lx, Lz = demo['Lx'], demo['Lz']
    q_pod = demo['q_pod']
    
    t_rom = rom_result['t']
    x_rom = rom_result['x']
    t_fo = fo['t']
    X_fo = fo['X']
    
    # Frame indices
    max_idx = min(len(t_rom) - 1, X_fo.shape[1] - 1)
    frame_indices = np.linspace(0, max_idx, n_frames, dtype=int)
    
    # Compute colorscale
    vmax_global = 0
    for idx in frame_indices[::10]:
        p_fo = X_fo[N:, idx]
        vmax_global = max(vmax_global, np.abs(p_fo).max())
    
    vmax_plot = vmax_global * 0.8
    err_max = vmax_plot * 0.1
    
    # Figure sizing
    domain_aspect = Lx / Lz
    plot_width = 12
    plot_height = plot_width / domain_aspect
    fig_height = 3 * plot_height + 2.5
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(plot_width + 2, fig_height))
    plt.subplots_adjust(bottom=0.12, hspace=0.25)
    
    # Initialize with frame 0
    p_full_0 = X_fo[N:, 0].reshape(Nx, Nz).T
    p_rom_0 = (Phi @ x_rom[0, :])[N:].reshape(Nx, Nz).T
    p_error_0 = p_full_0 - p_rom_0
    
    # Plot full-order
    im0 = axes[0].imshow(p_full_0, aspect='equal', cmap='RdBu_r',
                          vmin=-vmax_plot, vmax=vmax_plot,
                          extent=[0, Lx, Lz, 0])
    axes[0].set_ylabel('Depth (m)')
    plt.colorbar(im0, ax=axes[0], label='Pa', fraction=0.03, pad=0.02)
    title0 = axes[0].set_title('Full-order | t = 0 ms')
    
    # Plot ROM
    im1 = axes[1].imshow(p_rom_0, aspect='equal', cmap='RdBu_r',
                          vmin=-vmax_plot, vmax=vmax_plot,
                          extent=[0, Lx, Lz, 0])
    axes[1].set_ylabel('Depth (m)')
    plt.colorbar(im1, ax=axes[1], label='Pa', fraction=0.03, pad=0.02)
    title1 = axes[1].set_title(f'POD-ROM (q={q_pod}) | t = 0 ms')
    
    # Plot error
    im2 = axes[2].imshow(p_error_0, aspect='equal', cmap='RdBu_r',
                          vmin=-err_max, vmax=err_max,
                          extent=[0, Lx, Lz, 0])
    axes[2].set_xlabel('Range (m)')
    axes[2].set_ylabel('Depth (m)')
    plt.colorbar(im2, ax=axes[2], label='Pa', fraction=0.03, pad=0.02)
    title2 = axes[2].set_title('Spatial Error | 0.0%')
    
    # Add slider and play button
    ax_slider = plt.axes([0.15, 0.03, 0.5, 0.02])
    ax_button = plt.axes([0.7, 0.03, 0.1, 0.02])
    
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)
    button = Button(ax_button, 'Play')
    
    anim_running = [False]
    current_frame = [0]
    
    def update_frame(frame_num):
        frame_num = int(frame_num)
        idx = frame_indices[frame_num]
        
        # Match ROM index to full-order index
        rom_idx = int(idx * len(t_rom) / X_fo.shape[1])
        rom_idx = min(rom_idx, len(t_rom) - 1)
        
        t_ms = t_fo[idx] * 1000
        
        # Get fields
        p_full = X_fo[N:, idx].reshape(Nx, Nz).T
        p_rom = (Phi @ x_rom[rom_idx, :])[N:].reshape(Nx, Nz).T
        p_error = p_full - p_rom
        
        # Compute spatial error
        rel_err = np.linalg.norm(p_error) / (np.linalg.norm(p_full) + 1e-12) * 100
        
        # Update plots
        im0.set_data(p_full)
        im1.set_data(p_rom)
        im2.set_data(p_error)
        
        title0.set_text(f'Full-order | t = {t_ms:.0f} ms')
        title1.set_text(f'POD-ROM (q={q_pod}) | t = {t_ms:.0f} ms')
        title2.set_text(f'Spatial Error | {rel_err:.1f}%')
        
        fig.canvas.draw_idle()
    
    def on_slider_change(val):
        current_frame[0] = int(val)
        update_frame(val)
    
    def animate():
        if anim_running[0]:
            current_frame[0] = (current_frame[0] + 1) % n_frames
            slider.set_val(current_frame[0])
            timer.start(interval)
    
    def on_button_click(event):
        if anim_running[0]:
            anim_running[0] = False
            button.label.set_text('Play')
            timer.stop()
        else:
            anim_running[0] = True
            button.label.set_text('Pause')
            animate()
    
    slider.on_changed(on_slider_change)
    button.on_clicked(on_button_click)
    
    timer = fig.canvas.new_timer(interval=interval)
    timer.add_callback(animate)
    
    plt.suptitle(f"'{waveform}' | ROM q={q_pod} | Total Error: {err['rel_error']:.2f}%",
                 fontsize=14, y=0.98)
    plt.show()


# =============================================================================
# ANIMATE ALL WAVEFORMS (ROM left, Error right)
# =============================================================================
def animate_all_waveforms(demo, waveforms=None, n_frames=200, interval=50):
    """
    Animate specified waveforms: ROM pressure on left column, spatial error on right column.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    waveforms : list, optional
        List of waveform names to animate (default: all)
    n_frames : int
        Number of animation frames
    interval : int
        Animation interval in ms
    """
    from matplotlib.widgets import Slider, Button
    
    if waveforms is None:
        waveforms = list(WAVEFORMS.keys())
    
    # Validate
    for w in waveforms:
        if w not in WAVEFORMS:
            raise ValueError(f"Unknown waveform '{w}'. Options: {list(WAVEFORMS.keys())}")
        if w not in demo['full_order']:
            raise ValueError(f"No full-order reference for '{w}'")
        if demo['full_order'][w]['X'] is None:
            raise ValueError(f"No full-order snapshots (X_fo) saved for '{w}'. "
                            "Re-run demo_generate.py with save_snapshots=True.")
    
    n_waveforms = len(waveforms)
    
    # Simulate all ROMs
    print("Simulating ROMs...")
    rom_results = {}
    errors = {}
    for name in waveforms:
        rom_results[name] = simulate_rom(demo, name)
        errors[name] = compute_error(demo, rom_results[name])
        print(f"  {name}: {errors[name]['rel_error']:.2f}% error")
    
    # Extract parameters
    Phi = demo['Phi']
    N, Nx, Nz = demo['N'], demo['Nx'], demo['Nz']
    Lx, Lz = demo['Lx'], demo['Lz']
    q_pod = demo['q_pod']
    
    t_rom = rom_results[waveforms[0]]['t']
    X_fo_first = demo['full_order'][waveforms[0]]['X']
    max_idx = min(len(t_rom) - 1, X_fo_first.shape[1] - 1)
    frame_indices = np.linspace(0, max_idx, n_frames, dtype=int)
    
    # Compute global colorscale
    vmax_global = 0
    for name in waveforms:
        X_fo = demo['full_order'][name]['X']
        for idx in frame_indices[::20]:
            p_sample = X_fo[N:, idx]
            vmax_global = max(vmax_global, np.abs(p_sample).max())
    
    vmax_plot = vmax_global * 0.8
    err_max = vmax_plot * 0.15
    
    # Create figure: n_waveforms rows, 2 columns (ROM | Error)
    fig, axes = plt.subplots(n_waveforms, 2, figsize=(14, 3 * n_waveforms))
    plt.subplots_adjust(bottom=0.1, hspace=0.3, wspace=0.25)
    
    # Handle single waveform case
    if n_waveforms == 1:
        axes = axes.reshape(1, 2)
    
    # Initialize plots
    rom_images = []
    err_images = []
    rom_titles = []
    err_titles = []
    
    for i, name in enumerate(waveforms):
        x_rom = rom_results[name]['x']
        X_fo = demo['full_order'][name]['X']
        
        p_rom_0 = (Phi @ x_rom[0, :])[N:].reshape(Nx, Nz).T
        p_full_0 = X_fo[N:, 0].reshape(Nx, Nz).T
        p_error_0 = p_full_0 - p_rom_0
        
        # Left: ROM pressure field
        im_rom = axes[i, 0].imshow(p_rom_0, aspect='equal', cmap='RdBu_r',
                                    vmin=-vmax_plot, vmax=vmax_plot,
                                    extent=[0, Lx, Lz, 0])
        if i == 0:
            axes[i, 0].set_title('POD-ROM Pressure')
        axes[i, 0].set_ylabel(f"'{name}'\nDepth (m)")
        if i == n_waveforms - 1:
            axes[i, 0].set_xlabel('Range (m)')
        plt.colorbar(im_rom, ax=axes[i, 0], fraction=0.03, pad=0.02, label='Pa')
        
        err_val = errors[name]['rel_error']
        status = '✓' if err_val < 5 else '⚠' if err_val < 20 else '✗'
        title_rom = axes[i, 0].text(0.02, 0.98, f"{status} {err_val:.1f}% | t=0ms",
                                     transform=axes[i, 0].transAxes, fontsize=10,
                                     verticalalignment='top', color='white',
                                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Right: Spatial error
        im_err = axes[i, 1].imshow(p_error_0, aspect='equal', cmap='RdBu_r',
                                    vmin=-err_max, vmax=err_max,
                                    extent=[0, Lx, Lz, 0])
        if i == 0:
            axes[i, 1].set_title('Spatial Error (Full - ROM)')
        axes[i, 1].set_ylabel('Depth (m)')
        if i == n_waveforms - 1:
            axes[i, 1].set_xlabel('Range (m)')
        plt.colorbar(im_err, ax=axes[i, 1], fraction=0.03, pad=0.02, label='Pa')
        
        rel_err_0 = np.linalg.norm(p_error_0) / (np.linalg.norm(p_full_0) + 1e-12) * 100
        title_err = axes[i, 1].text(0.02, 0.98, f"Spatial: {rel_err_0:.1f}%",
                                     transform=axes[i, 1].transAxes, fontsize=10,
                                     verticalalignment='top', color='white',
                                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        rom_images.append(im_rom)
        err_images.append(im_err)
        rom_titles.append(title_rom)
        err_titles.append(title_err)
    
    # Slider and button
    ax_slider = plt.axes([0.15, 0.03, 0.5, 0.02])
    ax_button = plt.axes([0.7, 0.03, 0.08, 0.02])
    
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)
    button = Button(ax_button, 'Play')
    
    anim_running = [False]
    current_frame = [0]
    
    def update_frame(frame_num):
        frame_num = int(frame_num)
        idx = frame_indices[frame_num]
        t_ms = t_rom[idx] * 1000
        
        for i, name in enumerate(waveforms):
            x_rom = rom_results[name]['x']
            X_fo = demo['full_order'][name]['X']
            
            # Match indices
            rom_idx = int(idx * len(rom_results[name]['t']) / X_fo.shape[1])
            rom_idx = min(rom_idx, len(rom_results[name]['t']) - 1)
            fo_idx = min(idx, X_fo.shape[1] - 1)
            
            p_rom = (Phi @ x_rom[rom_idx, :])[N:].reshape(Nx, Nz).T
            p_full = X_fo[N:, fo_idx].reshape(Nx, Nz).T
            p_error = p_full - p_rom
            
            # Update images
            rom_images[i].set_data(p_rom)
            err_images[i].set_data(p_error)
            
            # Update titles
            err_val = errors[name]['rel_error']
            status = '✓' if err_val < 5 else '⚠' if err_val < 20 else '✗'
            rom_titles[i].set_text(f"{status} {err_val:.1f}% | t={t_ms:.0f}ms")
            
            rel_err = np.linalg.norm(p_error) / (np.linalg.norm(p_full) + 1e-12) * 100
            err_titles[i].set_text(f"Spatial: {rel_err:.1f}%")
        
        fig.canvas.draw_idle()
    
    def on_slider_change(val):
        current_frame[0] = int(val)
        update_frame(val)
    
    def animate():
        if anim_running[0]:
            current_frame[0] = (current_frame[0] + 1) % n_frames
            slider.set_val(current_frame[0])
            timer.start(interval)
    
    def on_button_click(event):
        if anim_running[0]:
            anim_running[0] = False
            button.label.set_text('Play')
            timer.stop()
        else:
            anim_running[0] = True
            button.label.set_text('Pause')
            animate()
    
    slider.on_changed(on_slider_change)
    button.on_clicked(on_button_click)
    
    timer = fig.canvas.new_timer(interval=interval)
    timer.add_callback(animate)
    
    plt.suptitle(f"ROM (q={q_pod}) vs Full-Order Comparison", fontsize=14, y=0.98)
    plt.show()

def animate_2_waveforms(demo, waveforms, n_frames=50, fps=10, error_type='relative', vmax_scale=0.3, figscale=1.0):
    """
    Animate exactly 2 waveforms: ROM pressure on left column, error on right column.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    waveforms : list
        List of exactly 2 waveform names to animate
    n_frames : int
        Number of animation frames
    fps : int
        Frames per second (controls animation speed)
    error_type : str
        'absolute' - signed error in Pa (Full - ROM)
        'relative' - percentage error at each point |Full - ROM| / |Full_max| * 100
    vmax_scale : float
        Scale factor for pressure colorbar (smaller = more saturated colors)
    figscale : float
        Scale factor for figure size (default 1.0)
    """
    from matplotlib.widgets import Slider, Button
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if len(waveforms) != 2:
        raise ValueError(f"Expected exactly 2 waveforms, got {len(waveforms)}")
    
    # Validate
    for w in waveforms:
        if w not in WAVEFORMS:
            raise ValueError(f"Unknown waveform '{w}'. Options: {list(WAVEFORMS.keys())}")
        if w not in demo['full_order']:
            raise ValueError(f"No full-order reference for '{w}'")
        if demo['full_order'][w]['X'] is None:
            raise ValueError(f"No full-order snapshots (X_fo) saved for '{w}'. "
                            "Re-run demo_generate.py with save_snapshots=True.")
    
    # Calculate interval from fps
    interval = int(1000 / fps)
    
    # Simulate ROMs
    print("Simulating ROMs...")
    rom_results = {}
    errors = {}
    for name in waveforms:
        rom_results[name] = simulate_rom(demo, name)
        errors[name] = compute_error(demo, rom_results[name])
        print(f"  {name}: {errors[name]['rel_error']:.2f}% error")
    
    # Compute speedup (simulation only)
    fo_time = demo['full_order'][waveforms[0]]['time']
    rom_time = rom_results[waveforms[0]]['elapsed']
    speedup_sim = fo_time / rom_time
    
    # Get training times (excluding setup)
    snapshot_time = demo.get('snapshot_time', None)
    svd_time = demo.get('svd_time', None)
    stability_time = demo.get('stability_time', None)
    build_time = demo.get('build_time', None)
    
    # Calculate training time without setup
    if all(t is not None for t in [snapshot_time, svd_time, stability_time, build_time]):
        training_time = snapshot_time + svd_time + stability_time + build_time
        has_timing_breakdown = True
    else:
        training_time = demo.get('training_time', None)
        has_timing_breakdown = False
    
    # Calculate breakeven
    if training_time is not None and fo_time > rom_time:
        breakeven_runs = int(np.ceil(training_time / (fo_time - rom_time)))
    else:
        breakeven_runs = None
    
    # Extract parameters
    Phi = demo['Phi']
    N, Nx, Nz = demo['N'], demo['Nx'], demo['Nz']
    Lx, Lz = demo['Lx'], demo['Lz']
    q_pod = demo['q_pod']
    
    t_rom = rom_results[waveforms[0]]['t']
    X_fo_first = demo['full_order'][waveforms[0]]['X']
    max_idx = min(len(t_rom) - 1, X_fo_first.shape[1] - 1)
    frame_indices = np.linspace(0, max_idx, n_frames, dtype=int)
    
    # Compute max field magnitude over all time for each waveform (for error normalization)
    max_norms = {}
    max_peaks = {}
    for name in waveforms:
        X_fo = demo['full_order'][name]['X']
        n_steps = X_fo.shape[1]
        max_norm = 0
        max_peak = 0
        for idx in range(0, n_steps, max(1, n_steps // 20)):
            p = X_fo[N:, idx]
            max_norm = max(max_norm, np.linalg.norm(p))
            max_peak = max(max_peak, np.abs(p).max())
        max_norms[name] = max_norm
        max_peaks[name] = max_peak
    
    # Compute global colorscales
    vmax_global = 0
    err_max_global = 0
    
    for name in waveforms:
        X_fo = demo['full_order'][name]['X']
        x_rom = rom_results[name]['x']
        
        for idx in frame_indices[::10]:
            p_full = X_fo[N:, idx]
            
            rom_idx = int(idx * len(rom_results[name]['t']) / X_fo.shape[1])
            rom_idx = min(rom_idx, len(rom_results[name]['t']) - 1)
            p_rom = (Phi @ x_rom[rom_idx, :])[N:]
            
            vmax_global = max(vmax_global, np.abs(p_full).max())
            
            if error_type == 'relative':
                err_field = np.abs(p_full - p_rom) / max_peaks[name] * 100
            else:
                err_field = np.abs(p_full - p_rom)
            
            err_max_global = max(err_max_global, err_field.max())
    
    # Clip pressure colorbar for stronger colors
    vmax_plot = vmax_global * vmax_scale
    
    # For relative error, cap at reasonable percentage
    if error_type == 'relative':
        err_max = min(err_max_global * 0.8, 50)
        err_label = 'Error (%)'
        err_cmap = 'magma'
    else:
        err_max = err_max_global * 0.5
        err_label = 'Error (Pa)'
        err_cmap = 'magma'
    
    # Figure sizing based on domain aspect ratio
    domain_aspect = Lx / Lz
    
    # Smaller base size
    plot_width = 7 * figscale
    plot_height = plot_width / domain_aspect
    
    # Figure dimensions
    fig_width = 2 * plot_width + 0.8 * figscale
    fig_height = 2 * (plot_height + 0.6 * figscale) + 2.0 * figscale
    
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    plt.subplots_adjust(bottom=0.08, top=0.86, left=0.08, right=0.98, hspace=0.4, wspace=0.15)
    
    # Initialize plots
    rom_images = []
    err_images = []
    rom_titles = []
    err_titles = []
    
    for i, name in enumerate(waveforms):
        x_rom = rom_results[name]['x']
        X_fo = demo['full_order'][name]['X']
        
        p_rom_0 = (Phi @ x_rom[0, :])[N:].reshape(Nx, Nz).T
        p_full_0 = X_fo[N:, 0].reshape(Nx, Nz).T
        
        if error_type == 'relative':
            p_error_0 = np.abs(p_full_0 - p_rom_0) / max_peaks[name] * 100
        else:
            p_error_0 = np.abs(p_full_0 - p_rom_0)
        
        # Left: ROM pressure field
        im_rom = axes[i, 0].imshow(p_rom_0, aspect='equal', cmap='RdBu_r',
                                    vmin=-vmax_plot, vmax=vmax_plot,
                                    extent=[0, Lx, Lz, 0])
        if i == 0:
            axes[i, 0].set_title('POD-ROM Pressure', fontsize=12 * figscale, fontweight='bold')
        axes[i, 0].set_ylabel(f"'{name}'\nDepth (m)", fontsize=10 * figscale, fontweight='bold')
        axes[i, 0].tick_params(labelsize=8 * figscale)
        
        # Colorbar below plot
        divider_rom = make_axes_locatable(axes[i, 0])
        cax_rom = divider_rom.append_axes("bottom", size="6%", pad=0.25)
        cbar_rom = plt.colorbar(im_rom, cax=cax_rom, orientation='horizontal')
        cbar_rom.set_label('Pa', fontsize=9 * figscale, fontweight='bold')
        cbar_rom.ax.tick_params(labelsize=8 * figscale)
        
        # Time only overlay
        title_rom = axes[i, 0].text(0.02, 0.96, f"t = 0 ms",
                                     transform=axes[i, 0].transAxes, fontsize=10 * figscale,
                                     fontweight='bold',
                                     verticalalignment='top', color='white',
                                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Right: Error field
        im_err = axes[i, 1].imshow(p_error_0, aspect='equal', cmap=err_cmap,
                                    vmin=0, vmax=err_max,
                                    extent=[0, Lx, Lz, 0])
        if i == 0:
            axes[i, 1].set_title(f'Spatial Error ({error_type.capitalize()})', fontsize=12 * figscale, fontweight='bold')
        axes[i, 1].set_ylabel('Depth (m)', fontsize=10 * figscale, fontweight='bold')
        axes[i, 1].tick_params(labelsize=8 * figscale)
        
        # Colorbar below plot
        divider_err = make_axes_locatable(axes[i, 1])
        cax_err = divider_err.append_axes("bottom", size="6%", pad=0.25)
        cbar_err = plt.colorbar(im_err, cax=cax_err, orientation='horizontal')
        cbar_err.set_label(err_label, fontsize=9 * figscale, fontweight='bold')
        cbar_err.ax.tick_params(labelsize=8 * figscale)
        
        # Compute spatial L2 error (normalized by max over all time)
        l2_err_0 = np.linalg.norm(p_full_0 - p_rom_0) / max_norms[name] * 100
        title_err = axes[i, 1].text(0.02, 0.96, f"L2: {l2_err_0:.2f}%",
                                     transform=axes[i, 1].transAxes, fontsize=9 * figscale,
                                     fontweight='bold',
                                     verticalalignment='top', color='white',
                                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        rom_images.append(im_rom)
        err_images.append(im_err)
        rom_titles.append(title_rom)
        err_titles.append(title_err)
    
    # Slider and button
    ax_slider = plt.axes([0.12, 0.02, 0.5, 0.015])
    ax_button = plt.axes([0.68, 0.02, 0.08, 0.02])
    
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)
    button = Button(ax_button, 'Play')
    
    anim_running = [False]
    current_frame = [0]
    
    def update_frame(frame_num):
        frame_num = int(frame_num)
        idx = frame_indices[frame_num]
        t_ms = t_rom[idx] * 1000
        
        for i, name in enumerate(waveforms):
            x_rom = rom_results[name]['x']
            X_fo = demo['full_order'][name]['X']
            
            # Match indices
            rom_idx = int(idx * len(rom_results[name]['t']) / X_fo.shape[1])
            rom_idx = min(rom_idx, len(rom_results[name]['t']) - 1)
            fo_idx = min(idx, X_fo.shape[1] - 1)
            
            p_rom = (Phi @ x_rom[rom_idx, :])[N:].reshape(Nx, Nz).T
            p_full = X_fo[N:, fo_idx].reshape(Nx, Nz).T
            
            if error_type == 'relative':
                p_error = np.abs(p_full - p_rom) / max_peaks[name] * 100
            else:
                p_error = np.abs(p_full - p_rom)
            
            # Update images
            rom_images[i].set_data(p_rom)
            err_images[i].set_data(p_error)
            
            # Update titles - time only for ROM plot
            rom_titles[i].set_text(f"t = {t_ms:.0f} ms")
            
            # L2 error only (normalized by max over all time)
            l2_err = np.linalg.norm(p_full - p_rom) / max_norms[name] * 100
            err_titles[i].set_text(f"L2: {l2_err:.2f}%")
        
        fig.canvas.draw_idle()
    
    def on_slider_change(val):
        current_frame[0] = int(val)
        update_frame(val)
    
    def animate():
        if anim_running[0]:
            current_frame[0] = (current_frame[0] + 1) % n_frames
            slider.set_val(current_frame[0])
            timer.start(interval)
    
    def on_button_click(event):
        if anim_running[0]:
            anim_running[0] = False
            button.label.set_text('Play')
            timer.stop()
        else:
            anim_running[0] = True
            button.label.set_text('Pause')
            animate()
    
    slider.on_changed(on_slider_change)
    button.on_clicked(on_button_click)
    
    timer = fig.canvas.new_timer(interval=interval)
    timer.add_callback(animate)
    
    # Build title with timing info
    fo_time_ms = fo_time * 1000
    rom_time_ms = rom_time * 1000
    
    # Main title line
    title_line1 = f"ROM (q={q_pod})  |  {speedup_sim:.0f}× speedup  |  FO: {fo_time_ms:.0f}ms → ROM: {rom_time_ms:.0f}ms"
    
    # Second line with training breakdown (including ROM simulation)
    if has_timing_breakdown:
        snapshot_ms = snapshot_time * 1000
        svd_ms = svd_time * 1000
        stability_ms = stability_time * 1000
        build_ms = build_time * 1000
        total_ms = snapshot_ms + svd_ms + stability_ms + build_ms + rom_time_ms
        
        title_line2 = f"Total: {total_ms:.0f}ms (snapshots: {snapshot_ms:.0f} + SVD: {svd_ms:.0f} + stability: {stability_ms:.0f} + build: {build_ms:.0f} + ROM: {rom_time_ms:.0f})"
        
        if breakeven_runs is not None:
            title_line2 += f"  |  Breakeven: {breakeven_runs} runs"
        
        full_title = f"{title_line1}\n{title_line2}"
    elif training_time is not None:
        training_ms = training_time * 1000
        total_ms = training_ms + rom_time_ms
        title_line2 = f"Total: {total_ms:.0f}ms (training: {training_ms:.0f} + ROM: {rom_time_ms:.0f})"
        if breakeven_runs is not None:
            title_line2 += f" (breakeven after {breakeven_runs} runs)"
        full_title = f"{title_line1}\n{title_line2}"
    else:
        full_title = title_line1
    
    plt.suptitle(full_title, fontsize=11 * figscale, fontweight='bold', y=0.98)
    plt.show()


def plot_2_error(demo, waveforms, figscale=1.0, separate=True):
    """
    Plot spatial L2 and max error over simulation time for exactly 2 waveforms.
    Both errors normalized by max field magnitude over all time.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    waveforms : list
        List of exactly 2 waveform names to plot
    figscale : float
        Scale factor for figure size
    separate : bool
        If True, each waveform gets its own row (2x2 grid)
        If False, both waveforms on same plots (1x2 grid)
    """
    if len(waveforms) != 2:
        raise ValueError(f"Expected exactly 2 waveforms, got {len(waveforms)}")
    
    # Validate
    for w in waveforms:
        if w not in WAVEFORMS:
            raise ValueError(f"Unknown waveform '{w}'. Options: {list(WAVEFORMS.keys())}")
        if w not in demo['full_order']:
            raise ValueError(f"No full-order reference for '{w}'")
        if demo['full_order'][w]['X'] is None:
            raise ValueError(f"No full-order snapshots (X_fo) saved for '{w}'.")
    
    # Simulate ROMs
    print("Simulating ROMs...")
    rom_results = {}
    for name in waveforms:
        rom_results[name] = simulate_rom(demo, name)
        print(f"  {name}: done")
    
    # Extract parameters
    Phi = demo['Phi']
    N = demo['N']
    
    colors = {'cont': 'green', 'pulse': 'blue', '5hz': 'red'}
    
    # Compute errors for each waveform
    all_errors = {}
    max_time = 0
    
    for name in waveforms:
        X_fo = demo['full_order'][name]['X']
        x_rom = rom_results[name]['x']
        t_rom = rom_results[name]['t']
        
        n_steps = X_fo.shape[1]
        t_fo = demo['full_order'][name]['t']
        
        # Find max field magnitude across entire simulation
        max_norm = 0
        max_peak = 0
        for idx in range(0, n_steps, max(1, n_steps // 20)):
            p = X_fo[N:, idx]
            max_norm = max(max_norm, np.linalg.norm(p))
            max_peak = max(max_peak, np.abs(p).max())
        
        # Compute spatial errors at each time step
        l2_errors = []
        max_errors = []
        times = []
        
        for idx in range(0, n_steps, max(1, n_steps // 200)):  # Sample ~200 points
            # Match ROM index to FO index
            rom_idx = int(idx * len(t_rom) / n_steps)
            rom_idx = min(rom_idx, len(t_rom) - 1)
            
            p_full = X_fo[N:, idx]
            p_rom = (Phi @ x_rom[rom_idx, :])[N:]
            
            # Normalize by max field magnitude over all time
            l2_err = np.linalg.norm(p_full - p_rom) / max_norm * 100
            max_err = np.abs(p_full - p_rom).max() / max_peak * 100
            
            l2_errors.append(l2_err)
            max_errors.append(max_err)
            times.append(t_fo[idx] * 1000)  # Convert to ms
        
        all_errors[name] = {'l2': l2_errors, 'max': max_errors, 'times': times}
        max_time = max(max_time, max(times))
    
    # Create figure based on layout
    if separate:
        # 2x2 grid - each waveform on its own row
        fig, axes = plt.subplots(2, 2, figsize=(14 * figscale, 8 * figscale))
        
        for i, name in enumerate(waveforms):
            color = colors.get(name, 'black')
            errs = all_errors[name]
            
            # Plot L2 error (left column)
            axes[i, 0].plot(errs['times'], errs['l2'], '-', color=color, lw=2)
            axes[i, 0].set_ylabel(f"'{name}'\nL2 Error (%)", fontsize=11 * figscale, fontweight='bold')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].tick_params(labelsize=10 * figscale)
            axes[i, 0].set_xlim([0, max_time])
            axes[i, 0].set_ylim(bottom=0)
            
            # Plot max error (right column)
            axes[i, 1].plot(errs['times'], errs['max'], '-', color=color, lw=2)
            axes[i, 1].set_ylabel(f"Max Error (%)", fontsize=11 * figscale, fontweight='bold')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].tick_params(labelsize=10 * figscale)
            axes[i, 1].set_xlim([0, max_time])
            axes[i, 1].set_ylim(bottom=0)
            
            # Only add x-label to bottom row
            if i == 1:
                axes[i, 0].set_xlabel('Time (ms)', fontsize=12 * figscale, fontweight='bold')
                axes[i, 1].set_xlabel('Time (ms)', fontsize=12 * figscale, fontweight='bold')
        
        # Column titles
        axes[0, 0].set_title('Spatial L2 Error Over Time', fontsize=14 * figscale, fontweight='bold')
        axes[0, 1].set_title('Max Pointwise Error Over Time', fontsize=14 * figscale, fontweight='bold')
    
    else:
        # 1x2 grid - both waveforms on same plots
        fig, axes = plt.subplots(1, 2, figsize=(14 * figscale, 5 * figscale))
        
        for name in waveforms:
            color = colors.get(name, 'black')
            errs = all_errors[name]
            
            # Plot L2 error
            axes[0].plot(errs['times'], errs['l2'], '-', color=color, lw=2, label=f"'{name}'")
            
            # Plot max error
            axes[1].plot(errs['times'], errs['max'], '-', color=color, lw=2, label=f"'{name}'")
        
        # Format L2 plot
        axes[0].set_xlabel('Time (ms)', fontsize=12 * figscale, fontweight='bold')
        axes[0].set_ylabel('L2 Error (%)', fontsize=12 * figscale, fontweight='bold')
        axes[0].set_title('Spatial L2 Error Over Time', fontsize=14 * figscale, fontweight='bold')
        axes[0].legend(fontsize=10 * figscale)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=10 * figscale)
        axes[0].set_xlim([0, max_time])
        axes[0].set_ylim(bottom=0)
        
        # Format max error plot
        axes[1].set_xlabel('Time (ms)', fontsize=12 * figscale, fontweight='bold')
        axes[1].set_ylabel('Max Error (%)', fontsize=12 * figscale, fontweight='bold')
        axes[1].set_title('Max Pointwise Error Over Time', fontsize=14 * figscale, fontweight='bold')
        axes[1].legend(fontsize=10 * figscale)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(labelsize=10 * figscale)
        axes[1].set_xlim([0, max_time])
        axes[1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nError Summary:")
    print("-" * 50)
    for name in waveforms:
        total_err = compute_error(demo, rom_results[name])['rel_error']
        print(f"  '{name}': Total L2 (hydrophone) = {total_err:.2f}%")


def animate_2_waveforms_error(demo, waveforms, n_frames=50, fps=10, error_type='relative', vmax_scale=0.3, figscale=1.0, save_gif=None):
    """
    Animate exactly 2 waveforms: ROM pressure, spatial error, and L2 error over time.
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    waveforms : list
        List of exactly 2 waveform names to animate
    n_frames : int
        Number of animation frames
    fps : int
        Frames per second (controls animation speed)
    error_type : str
        'absolute' - error in Pa (|Full - ROM|)
        'relative' - percentage error |Full - ROM| / max(|Full|) * 100
    vmax_scale : float
        Scale factor for pressure colorbar (smaller = more saturated colors)
    figscale : float
        Scale factor for figure size (default 1.0)
    save_gif : str or None
        If provided, save animation to this filepath (e.g., 'animation.gif')
        If None, show interactive animation
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if len(waveforms) != 2:
        raise ValueError(f"Expected exactly 2 waveforms, got {len(waveforms)}")
    
    # Validate
    for w in waveforms:
        if w not in WAVEFORMS:
            raise ValueError(f"Unknown waveform '{w}'. Options: {list(WAVEFORMS.keys())}")
        if w not in demo['full_order']:
            raise ValueError(f"No full-order reference for '{w}'")
        if demo['full_order'][w]['X'] is None:
            raise ValueError(f"No full-order snapshots (X_fo) saved for '{w}'. "
                            "Re-run demo_generate.py with save_snapshots=True.")
    
    # Calculate interval from fps
    interval = int(1000 / fps)
    
    # Simulate ROMs
    print("Simulating ROMs...")
    rom_results = {}
    errors = {}
    for name in waveforms:
        rom_results[name] = simulate_rom(demo, name)
        errors[name] = compute_error(demo, rom_results[name])
        print(f"  {name}: {errors[name]['rel_error']:.2f}% error")
    
    # Compute speedup (simulation only)
    fo_time = demo['full_order'][waveforms[0]]['time']
    rom_time = rom_results[waveforms[0]]['elapsed']
    speedup_sim = fo_time / rom_time
    
    # Get training times (excluding setup)
    snapshot_time = demo.get('snapshot_time', None)
    svd_time = demo.get('svd_time', None)
    stability_time = demo.get('stability_time', None)
    build_time = demo.get('build_time', None)
    
    # Calculate training time without setup
    if all(t is not None for t in [snapshot_time, svd_time, stability_time, build_time]):
        training_time = snapshot_time + svd_time + stability_time + build_time
        has_timing_breakdown = True
    else:
        training_time = demo.get('training_time', None)
        has_timing_breakdown = False
    
    # Calculate breakeven
    if training_time is not None and fo_time > rom_time:
        breakeven_runs = int(np.ceil(training_time / (fo_time - rom_time)))
    else:
        breakeven_runs = None
    
    # Extract parameters
    Phi = demo['Phi']
    N, Nx, Nz = demo['N'], demo['Nx'], demo['Nz']
    Lx, Lz = demo['Lx'], demo['Lz']
    q_pod = demo['q_pod']
    
    t_rom = rom_results[waveforms[0]]['t']
    X_fo_first = demo['full_order'][waveforms[0]]['X']
    max_idx = min(len(t_rom) - 1, X_fo_first.shape[1] - 1)
    frame_indices = np.linspace(0, max_idx, n_frames, dtype=int)
    
    # Compute max field magnitude over all time for each waveform (for error normalization)
    max_norms = {}
    max_peaks = {}
    for name in waveforms:
        X_fo = demo['full_order'][name]['X']
        n_steps = X_fo.shape[1]
        max_norm = 0
        max_peak = 0
        for idx in range(0, n_steps, max(1, n_steps // 20)):
            p = X_fo[N:, idx]
            max_norm = max(max_norm, np.linalg.norm(p))
            max_peak = max(max_peak, np.abs(p).max())
        max_norms[name] = max_norm
        max_peaks[name] = max_peak
    
    # Precompute L2 error over time for each waveform
    colors = {'cont': 'green', 'pulse': 'blue', '5hz': 'red'}
    l2_over_time = {}
    
    for name in waveforms:
        X_fo = demo['full_order'][name]['X']
        x_rom = rom_results[name]['x']
        t_rom_wf = rom_results[name]['t']
        
        n_steps = X_fo.shape[1]
        t_fo = demo['full_order'][name]['t']
        
        l2_errors = []
        times = []
        
        for idx in range(0, n_steps, max(1, n_steps // 200)):
            rom_idx = int(idx * len(t_rom_wf) / n_steps)
            rom_idx = min(rom_idx, len(t_rom_wf) - 1)
            
            p_full = X_fo[N:, idx]
            p_rom = (Phi @ x_rom[rom_idx, :])[N:]
            
            l2_err = np.linalg.norm(p_full - p_rom) / max_norms[name] * 100
            l2_errors.append(l2_err)
            times.append(t_fo[idx] * 1000)
        
        l2_over_time[name] = {'times': times, 'l2': l2_errors}
    
    # Compute global colorscales
    vmax_global = 0
    err_max_global = 0
    
    for name in waveforms:
        X_fo = demo['full_order'][name]['X']
        x_rom = rom_results[name]['x']
        
        for idx in frame_indices[::10]:
            p_full = X_fo[N:, idx]
            
            rom_idx = int(idx * len(rom_results[name]['t']) / X_fo.shape[1])
            rom_idx = min(rom_idx, len(rom_results[name]['t']) - 1)
            p_rom = (Phi @ x_rom[rom_idx, :])[N:]
            
            vmax_global = max(vmax_global, np.abs(p_full).max())
            
            if error_type == 'relative':
                err_field = np.abs(p_full - p_rom) / max_peaks[name] * 100
            else:
                err_field = np.abs(p_full - p_rom)
            
            err_max_global = max(err_max_global, err_field.max())
    
    # Clip pressure colorbar for stronger colors
    vmax_plot = vmax_global * vmax_scale
    
    # For relative error, cap at reasonable percentage
    if error_type == 'relative':
        err_max = min(err_max_global * 0.6, 50)
        err_label = 'Error (%)'
        err_cmap = 'magma'
    else:
        err_max = err_max_global * 0.5
        err_label = 'Error (Pa)'
        err_cmap = 'magma'
    
    # Figure sizing based on domain aspect ratio
    domain_aspect = Lx / Lz
    
    plot_width = 6 * figscale
    plot_height = plot_width / domain_aspect
    
    # Figure dimensions (3 rows, 2 columns) - all rows same height
    fig_width = 2 * plot_width + 1.0 * figscale
    fig_height = 3 * (plot_height + 0.8 * figscale) + 1.5 * figscale
    
    fig, axes = plt.subplots(3, 2, figsize=(fig_width, fig_height))
    plt.subplots_adjust(bottom=0.05, top=0.90, left=0.08, right=0.98, hspace=0.4, wspace=0.2)
    
    # Initialize plots
    rom_images = []
    err_images = []
    time_lines = []
    time_dots = []
    
    # Get max time for x-axis
    max_time = max(l2_over_time[waveforms[0]]['times'][-1], l2_over_time[waveforms[1]]['times'][-1])
    
    # Get max L2 for y-axis (shared across both)
    max_l2 = max(max(l2_over_time[waveforms[0]]['l2']), max(l2_over_time[waveforms[1]]['l2'])) * 1.1
    
    for j, name in enumerate(waveforms):
        x_rom = rom_results[name]['x']
        X_fo = demo['full_order'][name]['X']
        
        p_rom_0 = (Phi @ x_rom[0, :])[N:].reshape(Nx, Nz).T
        p_full_0 = X_fo[N:, 0].reshape(Nx, Nz).T
        
        if error_type == 'relative':
            p_error_0 = np.abs(p_full_0 - p_rom_0) / max_peaks[name] * 100
        else:
            p_error_0 = np.abs(p_full_0 - p_rom_0)
        
        color = colors.get(name, 'black')
        
        # Row 0: ROM pressure field
        im_rom = axes[0, j].imshow(p_rom_0, aspect='equal', cmap='RdBu_r',
                                    vmin=-vmax_plot, vmax=vmax_plot,
                                    extent=[0, Lx, Lz, 0])
        axes[0, j].set_title(f"'{name}'", fontsize=12 * figscale, fontweight='bold')
        if j == 0:
            axes[0, j].set_ylabel('POD-ROM\nDepth (m)', fontsize=10 * figscale, fontweight='bold')
        axes[0, j].tick_params(labelsize=8 * figscale)
        
        divider_rom = make_axes_locatable(axes[0, j])
        cax_rom = divider_rom.append_axes("bottom", size="6%", pad=0.25)
        cbar_rom = plt.colorbar(im_rom, cax=cax_rom, orientation='horizontal')
        cbar_rom.set_label('Pressure (Pa)', fontsize=8 * figscale, fontweight='bold')
        cbar_rom.ax.tick_params(labelsize=7 * figscale)
        
        # Row 1: Spatial error field
        im_err = axes[1, j].imshow(p_error_0, aspect='equal', cmap=err_cmap,
                                    vmin=0, vmax=err_max,
                                    extent=[0, Lx, Lz, 0])
        if j == 0:
            axes[1, j].set_ylabel('Pointwise Error\nDepth (m)', fontsize=10 * figscale, fontweight='bold')
        axes[1, j].tick_params(labelsize=8 * figscale)
        
        divider_err = make_axes_locatable(axes[1, j])
        cax_err = divider_err.append_axes("bottom", size="6%", pad=0.25)
        cbar_err = plt.colorbar(im_err, cax=cax_err, orientation='horizontal')
        cbar_err.set_label(err_label, fontsize=8 * figscale, fontweight='bold')
        cbar_err.ax.tick_params(labelsize=7 * figscale)
        
        # Row 2: L2 error over time (match aspect ratio of other plots)
        l2_err_0 = np.linalg.norm(p_full_0 - p_rom_0) / max_norms[name] * 100
        
        axes[2, j].set_box_aspect(Lz / Lx)  # Match domain aspect ratio
        axes[2, j].plot(l2_over_time[name]['times'], l2_over_time[name]['l2'], '-', color=color, lw=2)
        time_line = axes[2, j].axvline(x=0, color='red', lw=1.5, linestyle='--')
        time_dot, = axes[2, j].plot([0], [l2_err_0], 'o', color='red', markersize=8)
        
        if j == 0:
            axes[2, j].set_ylabel('L2 Error (%)', fontsize=10 * figscale, fontweight='bold')
        axes[2, j].set_xlabel('Time (ms)', fontsize=10 * figscale, fontweight='bold')
        axes[2, j].set_xlim([0, max_time])
        axes[2, j].set_ylim([0, max_l2])
        axes[2, j].grid(True, alpha=0.3)
        axes[2, j].tick_params(labelsize=8 * figscale)
        
        rom_images.append(im_rom)
        err_images.append(im_err)
        time_lines.append(time_line)
        time_dots.append(time_dot)
    
    # Build title with timing info
    fo_time_ms = fo_time * 1000
    rom_time_ms = rom_time * 1000
    
    title_line1 = f"ROM (q={q_pod})  |  {speedup_sim:.0f}× speedup  |  FO: {fo_time_ms:.0f}ms → ROM: {rom_time_ms:.0f}ms"
    
    if has_timing_breakdown:
        snapshot_ms = snapshot_time * 1000
        svd_ms = svd_time * 1000
        stability_ms = stability_time * 1000
        build_ms = build_time * 1000
        total_ms = snapshot_ms + svd_ms + stability_ms + build_ms + rom_time_ms
        
        title_line2 = f"Total: {total_ms:.0f}ms (snapshots: {snapshot_ms:.0f} + SVD: {svd_ms:.0f} + stability: {stability_ms:.0f} + build: {build_ms:.0f} + ROM: {rom_time_ms:.0f})"
        
        if breakeven_runs is not None:
            title_line2 += f"  |  Breakeven: {breakeven_runs} runs"
        
        full_title = f"{title_line1}\n{title_line2}"
    elif training_time is not None:
        training_ms = training_time * 1000
        total_ms = training_ms + rom_time_ms
        title_line2 = f"Total: {total_ms:.0f}ms (training: {training_ms:.0f} + ROM: {rom_time_ms:.0f})"
        if breakeven_runs is not None:
            title_line2 += f" (breakeven after {breakeven_runs} runs)"
        full_title = f"{title_line1}\n{title_line2}"
    else:
        full_title = title_line1
    
    plt.suptitle(full_title, fontsize=10 * figscale, fontweight='bold', y=0.98)
    
    # Animation update function
    def update(frame_num):
        idx = frame_indices[frame_num]
        t_ms = t_rom[idx] * 1000
        
        for j, name in enumerate(waveforms):
            x_rom = rom_results[name]['x']
            X_fo = demo['full_order'][name]['X']
            
            rom_idx = int(idx * len(rom_results[name]['t']) / X_fo.shape[1])
            rom_idx = min(rom_idx, len(rom_results[name]['t']) - 1)
            fo_idx = min(idx, X_fo.shape[1] - 1)
            
            p_rom = (Phi @ x_rom[rom_idx, :])[N:].reshape(Nx, Nz).T
            p_full = X_fo[N:, fo_idx].reshape(Nx, Nz).T
            
            if error_type == 'relative':
                p_error = np.abs(p_full - p_rom) / max_peaks[name] * 100
            else:
                p_error = np.abs(p_full - p_rom)
            
            # Update images
            rom_images[j].set_data(p_rom)
            err_images[j].set_data(p_error)
            
            # Update time marker on L2 plot
            l2_err = np.linalg.norm(p_full - p_rom) / max_norms[name] * 100
            time_lines[j].set_xdata([t_ms, t_ms])
            time_dots[j].set_data([t_ms], [l2_err])
        
        return rom_images + err_images + time_lines + time_dots
    
    # Save as GIF or show interactive
    if save_gif is not None:
        from matplotlib.animation import FuncAnimation
        
        print(f"Generating animation with {n_frames} frames...")
        anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
        
        print(f"Saving to {save_gif}...")
        anim.save(save_gif, writer='pillow', fps=fps)
        print(f"Saved: {save_gif}")
        plt.close(fig)
    else:
        # Interactive mode with slider and button
        from matplotlib.widgets import Slider, Button
        
        ax_slider = plt.axes([0.10, 0.02, 0.5, 0.012])
        ax_button = plt.axes([0.65, 0.02, 0.08, 0.02])
        
        slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)
        button = Button(ax_button, 'Play')
        
        anim_running = [False]
        current_frame = [0]
        
        def on_slider_change(val):
            current_frame[0] = int(val)
            update(int(val))
            fig.canvas.draw_idle()
        
        def animate():
            if anim_running[0]:
                current_frame[0] = (current_frame[0] + 1) % n_frames
                slider.set_val(current_frame[0])
                timer.start(interval)
        
        def on_button_click(event):
            if anim_running[0]:
                anim_running[0] = False
                button.label.set_text('Play')
                timer.stop()
            else:
                anim_running[0] = True
                button.label.set_text('Pause')
                animate()
        
        slider.on_changed(on_slider_change)
        button.on_clicked(on_button_click)
        
        timer = fig.canvas.new_timer(interval=interval)
        timer.add_callback(animate)
        
        plt.show()