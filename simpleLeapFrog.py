import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

def LeapfrogSolver(eval_f, x_start, p, eval_u, NumIter, dt, visualize=False, gif_file_name="Leapfrog_visualization.gif"):
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
    
    # Initial condition
    X[:, 0] = np.reshape(x_start, [-1])
    t[0] = 0
    
    # First step: Use Forward Euler to bootstrap (or RK2 for better accuracy)
    print("Bootstrapping with first step...")
    u0 = eval_u(t[0])
    f0 = eval_f(np.reshape(X[:, 0], [-1, 1]), p, u0)
    
    # Option 1: Simple Forward Euler for first step
    # X[:, 1] = X[:, 0] + dt * f0.reshape(X[:, 0].shape)
    
    # Option 2: RK2 for better accuracy in bootstrap
    k1 = f0.reshape(X[:, 0].shape)
    X_mid = X[:, 0] + 0.5 * dt * k1
    u_mid = eval_u(t[0] + 0.5*dt)
    f_mid = eval_f(np.reshape(X_mid, [-1, 1]), p, u_mid)
    k2 = f_mid.reshape(X[:, 0].shape)
    X[:, 1] = X[:, 0] + dt * k2
    t[1] = dt
    
    # Main leapfrog loop
    print(f"Running {NumIter-1} leapfrog steps...")
    for n in range(1, NumIter):
        if n % max(1, NumIter//10) == 0:
            print(f"  Progress: {100*n/NumIter:.1f}%")
        
        t[n+1] = t[n] + dt
        u_n = eval_u(t[n])
        f_n = eval_f(np.reshape(X[:, n], [-1, 1]), p, u_n)
        
        # Leapfrog update: x_{n+1} = x_{n-1} + 2*dt*f(x_n)
        X[:, n+1] = X[:, n-1] + 2 * dt * f_n.reshape(X[:, n].shape)
        
        # Optional: Add artificial damping for long-time stability
        # This slightly breaks symplecticity but prevents drift
        damping_factor = 0.0001  # Very small
        X[:, n+1] = X[:, n+1] - damping_factor * (X[:, n+1] - X[:, n])
    
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