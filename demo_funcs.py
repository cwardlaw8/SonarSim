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
T_MAX = 0.8  # 800 ms


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
    
    Returns
    -------
    fig : matplotlib figure
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
    return fig


def plot_all_inputs(t_max=T_MAX, n_points=1000):
    """
    Plot all waveforms and their frequency content.
    
    Returns
    -------
    fig : matplotlib figure
    """
    waveform_names = list(WAVEFORMS.keys())
    n_waveforms = len(waveform_names)
    
    fig, axes = plt.subplots(n_waveforms, 2, figsize=(12, 3 * n_waveforms))
    
    t = np.linspace(0, t_max, n_points)
    dt = t[1] - t[0]
    freqs = np.fft.rfftfreq(n_points, dt)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_waveforms))
    
    for i, name in enumerate(waveform_names):
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
    return fig


# =============================================================================
# ANIMATE COMPARISON (Full-order, ROM, Error)
# =============================================================================

def animate_comparison(demo, waveform, n_frames=200, interval=50):
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
    
    return fig


# =============================================================================
# ANIMATE ALL WAVEFORMS SIDE BY SIDE
# =============================================================================

def animate_all_waveforms(demo, n_frames=200, interval=50):
    """
    Animate all waveforms side by side (ROM only).
    
    Parameters
    ----------
    demo : dict
        Output from load_demo_data()
    n_frames : int
        Number of animation frames
    interval : int
        Animation interval in ms
    """
    from matplotlib.widgets import Slider, Button
    
    waveform_names = list(WAVEFORMS.keys())
    n_waveforms = len(waveform_names)
    
    # Simulate all ROMs
    print("Simulating ROMs...")
    rom_results = {}
    errors = {}
    for name in waveform_names:
        rom_results[name] = simulate_rom(demo, name)
        errors[name] = compute_error(demo, rom_results[name])
        print(f"  {name}: {errors[name]['rel_error']:.2f}% error")
    
    # Extract parameters
    Phi = demo['Phi']
    N, Nx, Nz = demo['N'], demo['Nx'], demo['Nz']
    Lx, Lz = demo['Lx'], demo['Lz']
    q_pod = demo['q_pod']
    
    t_rom = rom_results['cont']['t']
    max_idx = len(t_rom) - 1
    frame_indices = np.linspace(0, max_idx, n_frames, dtype=int)
    
    # Compute global colorscale
    vmax_global = 0
    for name in waveform_names:
        x_rom = rom_results[name]['x']
        for idx in frame_indices[::20]:
            p_sample = (Phi @ x_rom[idx, :])[N:]
            vmax_global = max(vmax_global, np.abs(p_sample).max())
    
    vmax_plot = vmax_global * 0.8
    
    # Create figure
    fig, axes = plt.subplots(1, n_waveforms, figsize=(5 * n_waveforms, 5))
    plt.subplots_adjust(bottom=0.18, wspace=0.3)
    
    # Initialize plots
    images = []
    titles = []
    
    for i, name in enumerate(waveform_names):
        x_rom = rom_results[name]['x']
        p_rom_0 = (Phi @ x_rom[0, :])[N:].reshape(Nx, Nz).T
        
        im = axes[i].imshow(p_rom_0, aspect='equal', cmap='RdBu_r',
                            vmin=-vmax_plot, vmax=vmax_plot,
                            extent=[0, Lx, Lz, 0])
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='Pa')
        axes[i].set_xlabel('Range (m)')
        if i == 0:
            axes[i].set_ylabel('Depth (m)')
        
        err_val = errors[name]['rel_error']
        status = '✓' if err_val < 5 else '⚠' if err_val < 20 else '✗'
        title = axes[i].set_title(f"'{name}'\n{status} {err_val:.1f}% | t=0ms", fontsize=10)
        
        images.append(im)
        titles.append(title)
    
    # Slider and button
    ax_slider = plt.axes([0.15, 0.06, 0.5, 0.03])
    ax_button = plt.axes([0.7, 0.06, 0.1, 0.03])
    
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)
    button = Button(ax_button, 'Play')
    
    anim_running = [False]
    current_frame = [0]
    
    def update_frame(frame_num):
        frame_num = int(frame_num)
        idx = frame_indices[frame_num]
        t_ms = t_rom[idx] * 1000
        
        for i, name in enumerate(waveform_names):
            x_rom = rom_results[name]['x']
            p_rom = (Phi @ x_rom[idx, :])[N:].reshape(Nx, Nz).T
            images[i].set_data(p_rom)
            
            err_val = errors[name]['rel_error']
            status = '✓' if err_val < 5 else '⚠' if err_val < 20 else '✗'
            titles[i].set_text(f"'{name}'\n{status} {err_val:.1f}% | t={t_ms:.0f}ms")
        
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
    
    plt.suptitle(f"ROM (q={q_pod}) Response to Different Waveforms", fontsize=14, y=0.98)
    plt.show()
    
    return fig