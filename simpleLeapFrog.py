import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from functools import partial

def LeapfrogSolver(eval_f, x_start, p, eval_u, NumIter, dt, visualize=False, gif_file_name="Leapfrog_visualization.gif", verbose=True):
    """
    Leapfrog integration scheme for second-order wave equations.
    Much more stable than Forward Euler for oscillatory systems.
    
    The leapfrog scheme uses centered differences in time:
    x_{n+1} = x_{n-1} + 2*dt*f(x_n, t_n)
    
    This is symplectic (energy-conserving) for wave equations.
    
    Parameters:
    -----------
    eval_f : function
        Function that evaluates dx/dt = f(x, p, u)
    x_start : array
        Initial state vector [pressure; velocity]
    p : dict
        Parameters including system matrices
    eval_u : function
        Source function u(t)
    NumIter : int
        Number of time steps
    dt : float
        Time step (use dt instead of w for clarity)
    visualize : bool
        Whether to create animation
    gif_file_name : str
        Output filename for animation
    
    Returns:
    --------
    X : array
        State history (shape: [n_states, n_timesteps])
    t : array
        Time array
    """
    
    print("Running Leapfrog solver (stable for wave equations)...")
    
    NumIter = int(NumIter)
    n_states = len(x_start)
    
    # Allocate storage
    X = np.zeros((n_states, NumIter + 1))
    t = np.zeros(NumIter + 1)
    
    # Initial condition at t_start (handle if provided in p dict)
    t_start = p.get('t_start', 0.0) if isinstance(p, dict) else 0.0
    X[:, 0] = np.reshape(x_start, [-1])
    t[0] = t_start
    
    # Bootstrap first step using RK4 for O(dt^4) accuracy
    # This ensures X[:, 1] is as accurate as the main loop
    if verbose:
        print("Bootstrapping with RK4 for first step...")
    
    u0 = eval_u(t[0])
    x0 = X[:, 0].reshape(-1, 1)
    
    # RK4 coefficients
    k1 = eval_f(x0, p, u0).reshape(-1)
    
    u_half = eval_u(t[0] + 0.5*dt)
    x_k2 = (x0 + 0.5*dt*k1.reshape(-1, 1))
    k2 = eval_f(x_k2, p, u_half).reshape(-1)
    
    x_k3 = (x0 + 0.5*dt*k2.reshape(-1, 1))
    k3 = eval_f(x_k3, p, u_half).reshape(-1)
    
    u1 = eval_u(t[0] + dt)
    x_k4 = (x0 + dt*k3.reshape(-1, 1))
    k4 = eval_f(x_k4, p, u1).reshape(-1)
    
    # RK4 update
    X[:, 1] = X[:, 0] + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    t[1] = t_start + dt
    
    # Main leapfrog loop
    if verbose:
        print(f"Running {NumIter-1} leapfrog steps...")
    for n in range(1, NumIter):
        if n % max(1, NumIter//10) == 0 and verbose:
            print(f"  Progress: {100*n/NumIter:.1f}%")
        
        t[n+1] = t[n] + dt
        u_n = eval_u(t[n])
        f_n = eval_f(np.reshape(X[:, n], [-1, 1]), p, u_n)
        
        # Pure Leapfrog update: x_{n+1} = x_{n-1} + 2*dt*f(x_n)
        # This is symplectic (energy-conserving) - do NOT add artificial damping
        X[:, n+1] = X[:, n-1] + 2 * dt * f_n.reshape(X[:, n].shape)
    
    if verbose:
        print("Leapfrog integration complete!")
    
    # Optional visualization
    if visualize:
        from VisualizeState import VisualizeState
        
        if X.shape[0] > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax = (ax1, ax2)
        else:
            fig, ax = plt.subplots(1, 1)
            ax = (ax,)
        
        plt.tight_layout(pad=3.0)
        ani = animation.FuncAnimation(
            fig,
            partial(VisualizeState, t=t, X=X, ax=ax),
            frames=NumIter + 1,
            repeat=False,
            interval=100
        )
        
        ani.save(gif_file_name, writer="pillow")
        plt.show()
    
    return X, t


def test_solver_stability(p, eval_f, eval_u, x_start, max_dt_FE):
    """
    Compare Forward Euler vs Leapfrog stability for your system.
    """
    
    print("\n" + "="*60)
    print("SOLVER STABILITY COMPARISON")
    print("="*60)
    
    # Short test duration
    test_duration = 0.5  # seconds
    dt = max_dt_FE * 0.5
    num_iter = int(test_duration / dt)
    
    # Run both solvers
    print("\n1. Testing SimpleSolver (Forward Euler)...")
    from SimpleSolver import SimpleSolver
    X_euler, t_euler = SimpleSolver(eval_f, x_start, p, eval_u, 
                                    num_iter, dt, visualize=False)
    
    print("\n2. Testing LeapfrogSolver...")
    X_leap, t_leap = LeapfrogSolver(eval_f, x_start, p, eval_u, 
                                    num_iter, dt, visualize=False)
    
    # Compare energy over time
    N = p['Nx'] * p['Nz']
    energy_euler = np.sum(X_euler[:N, :]**2, axis=0)  # Pressure energy
    energy_leap = np.sum(X_leap[:N, :]**2, axis=0)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t_euler*1000, energy_euler, 'r-', label='Forward Euler', linewidth=2)
    ax1.plot(t_leap*1000, energy_leap, 'b-', label='Leapfrog', linewidth=2)
    ax1.set_ylabel('Energy (Pressure²)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Energy Evolution: Forward Euler vs Leapfrog')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot growth rate
    growth_euler = energy_euler / (energy_euler[0] + 1e-10)
    growth_leap = energy_leap / (energy_leap[0] + 1e-10)
    
    ax2.plot(t_euler*1000, growth_euler, 'r-', label='Forward Euler', linewidth=2)
    ax2.plot(t_leap*1000, growth_leap, 'b-', label='Leapfrog', linewidth=2)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Relative Energy (E/E₀)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('Energy Growth Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Forward Euler - Final energy ratio: {growth_euler[-1]:.2e}")
    print(f"Leapfrog      - Final energy ratio: {growth_leap[-1]:.2e}")
    
    if growth_euler[-1] > 1.1:
        print("⚠️  Forward Euler is UNSTABLE (growing energy)")
    if abs(growth_leap[-1] - 1.0) < 0.1:
        print("✓  Leapfrog is STABLE (conserved energy)")
    
    return fig

def test_dt_sweep_leapfrog(eval_f, x_start, p, eval_u,
                           exponents,
                           max_steps,
                           use_outputs=False,
                           eval_g=None,
                           quiet=False):
    """
    Cold-restart Δt sweep for Leapfrog (no solver changes).
    - For each Δt = 10**e (e in `exponents`), we define a common horizon T from the coarsest step
      and run Leapfrog in chunks of at most `max_steps` steps, re-bootstrapping each chunk (cold restart).
    - The last chunk length is chosen so that each run lands exactly at T (constant adjusted Δt per run).
    - We compare the final vector (state or outputs) against the smallest feasible Δt run and plot a V-curve.

    Parameters
    ----------
    eval_f : callable
        f(x, p, u) -> dx/dt  (expects x as vector/column; your LeapfrogSolver calls it)
    x_start : ndarray
        Initial state at the start time; dtype sets precision.
    p : dict
        Parameter dict; if it contains 'A', we estimate a κ(A)·eps floor line.
        If it contains 't_start', we advance it per chunk so u(t) stays aligned.
    eval_u : callable
        u(t) -> scalar input (absolute time). We wrap it with a time offset per chunk.
    exponents : iterable[int]
        Δt candidates: Δt = 10**e, ordered from coarse to fine (e.g., range(-4, -13, -1)).
    max_steps : int
        Hard cap per LeapfrogSolver call. Larger runs are split into several cold-restart chunks.
    use_outputs : bool
        If True (and eval_g provided), compare measurement y=g(x) at final time; else compare state x.
    eval_g : callable or None
        g(x, p) -> measurement vector (e.g., hydrophones). Used only if use_outputs=True.
    quiet : bool
        If False, prints progress for each Δt and per-chunk progress.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Semilogy plot of relative error vs -log10(Δt_adj).
    """

    # ---------- Define shared horizon T from the coarsest Δt ----------
    exponents = list(exponents)
    if len(exponents) < 2:
        raise ValueError("Provide at least two exponents to produce a sweep.")
    dt0 = 10.0 ** float(exponents[0])               # coarsest step
    T   = max_steps * dt0                           # common horizon so the coarsest run fits in 1 chunk
    if not quiet:
        print(f"[sweep] Coarsest Δt={dt0:.1e} -> T={T:.3e} with max_steps={max_steps:,}")

    # ---------- Floors/guides ----------
    x0_arr = np.asarray(x_start).reshape(-1)
    eps = np.finfo(x0_arr.dtype).eps if x0_arr.dtype.kind == 'f' else np.finfo(np.float64).eps
    kappa_line = None
    if isinstance(p, dict) and ('A' in p):
        try:
            kappa = float(np.linalg.cond(p['A']))
            if np.isfinite(kappa) and kappa > 0:
                kappa_line = kappa * eps
        except Exception:
            kappa_line = None

    # ---------- One full run at fixed (adjusted) Δt using cold-restart chunking ----------
    def _run_cold_restart(dt_target):
        """
        Run from t=0 to T using constant adjusted Δt (Δt_adj = T / steps_total),
        split into chunks of <= max_steps. Each chunk uses LeapfrogSolver fresh (RK2 bootstrap).
        Returns (final_vec, dt_adj, steps_total).
        """
        # total steps and adjusted dt so we land exactly at T
        steps_total = int(np.ceil(T / dt_target))
        dt_adj = T / steps_total

        # running state/time and a working copy of p to move t_start forward (if it exists)
        x_curr = x0_arr.copy()
        t_offset = float(p.get('t_start', 0.0))  # absolute time anchor
        remaining = steps_total

        # Wrapper to shift time for u(t) per chunk
        def make_u_with_offset(offset):
            return (lambda t: eval_u(t + offset))

        # Process chunks
        chunk_idx = 0
        while remaining > 0:
            steps = min(max_steps, remaining)
            # build a shallow copy of p with an updated t_start (helps if solver uses it)
            p_chunk = dict(p)
            p_chunk['t_start'] = t_offset

            # shifted input for this chunk
            eval_u_shifted = make_u_with_offset(t_offset)

            # Call your LeapfrogSolver correctly: (eval_f, x_start, p, eval_u, NumIter, w)
            out = LeapfrogSolver(eval_f, x_curr, p_chunk, eval_u_shifted, steps, dt_adj, verbose=False)

            # Extract final state from the chunk
            if isinstance(out, tuple):
                X_chunk = out[0]
                x_curr = (X_chunk[:, -1] if getattr(X_chunk, "ndim", 1) == 2
                          else np.asarray(X_chunk).reshape(-1))
            else:
                x_curr = np.asarray(out).reshape(-1)

            # advance absolute time and remaining steps
            t_offset += steps * dt_adj
            remaining -= steps
            chunk_idx += 1

            if not quiet:
                print(f"   [Δt={dt_adj:.1e}] chunk {chunk_idx} done "
                      f"(steps={steps:,}, t={t_offset:.3e}/{T:.3e})")

        # Return state or outputs as comparison vector
        if use_outputs and (eval_g is not None):
            vec = np.asarray(eval_g(x_curr, p), dtype=float).reshape(-1)
        else:
            vec = x_curr.reshape(-1)
        return vec, dt_adj, steps_total

    # ---------- Build runs across Δt candidates ----------
    records = []
    for e in exponents:
        dt = 10.0 ** float(e)
        if not quiet:
            print(f"[run] Δt_target={dt:.1e} → cold-restart chunks up to max_steps={max_steps:,}")
        vec, dt_adj, steps_total = _run_cold_restart(dt)
        records.append((dt, dt_adj, steps_total, vec))

    # Reference = smallest Δt (last entry assuming exponents in descending order)
    dt_ref, dt_ref_adj, steps_ref, vec_ref = records[-1]
    ref_norm = np.linalg.norm(vec_ref) + 1e-300

    # ---------- Compute errors ----------
    digits, errors = [], []
    for dt, dt_adj, steps_total, vec in records:
        rel_err = np.linalg.norm(vec - vec_ref) / ref_norm
        digits.append(-np.log10(dt_adj))
        errors.append(rel_err)
    digits = np.asarray(digits)
    errors = np.asarray(errors)

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.semilogy(digits, errors, marker='o', linewidth=1.5)
    ax.set_xlabel(r"digits in $\Delta t$  ( $-\log_{10} \Delta t_{\mathrm{adj}}$ )")
    ax.set_ylabel("relative error vs smallest Δt")
    ax.set_title("Leapfrog Δt sweep (cold-restart chunking)")
    ax.grid(True, which='both', alpha=0.3)

    # Floors
    ax.axhline(eps, linestyle='--', linewidth=1.0, label=f"machine eps ≈ {eps:.1e}")
    if kappa_line is not None:
        ax.axhline(kappa_line, linestyle=':', linewidth=1.0, label=f"κ(A)·eps ≈ {kappa_line:.1e}")
    ax.legend()

    # Console summary
    if not quiet:
        print(f"[summary] dtype={x0_arr.dtype}, eps={eps:.1e}")
        if kappa_line is not None:
            print(f"[summary] κ(A)·eps ≈ {kappa_line:.2e}")
        print("[summary] Runs included:")
        for idx, (dt, dt_adj, steps_total, _) in enumerate(records):
            print(f"[{idx}] Δt_target={dt:.1e}  Δt_adj={dt_adj:.2e}  total_steps={steps_total:,}, rel_err={errors[idx]:.2e}")

    return fig