import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_wave_animation(X, t, p, save_filename='wave_animation.gif'):
    """
    Create an animation of wave propagation in the 2D container
    
    Parameters:
    X: state matrix (2*Nx*Nz, num_timesteps)
    t: time vector
    p: parameter dictionary containing grid info
    save_filename: name for the saved animation
    """
    Nx, Nz = p['Nx'], p['Nz']
    dx, dz = p['dx'], p['dz']
    
    # Extract pressure field (first half of state vector)
    N = Nx * Nz
    pressure_history = X[:N, :].reshape(Nx, Nz, -1)
    
    # Create coordinate grids for plotting
    x_coords = np.arange(Nx) * dx
    z_coords = np.arange(Nz) * dz
    X_grid, Z_grid = np.meshgrid(x_coords, z_coords, indexing='ij')
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find global min/max for consistent color scaling
    vmin = np.min(pressure_history)
    vmax = np.max(pressure_history)
    v_abs_max = max(abs(vmin), abs(vmax))
    
    # Create initial plot
    im = ax.imshow(pressure_history[:, :, 0].T, 
                   extent=[0, Nx*dx, 0, Nz*dz],
                   origin='lower', 
                   cmap='RdBu_r', 
                   vmin=-v_abs_max, 
                   vmax=v_abs_max,
                   aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Pressure (Pa)')
    
    # Mark the sonar source location
    sonar_x = p['sonar_ix'] * dx
    sonar_z = p['sonar_iz'] * dz
    ax.plot(sonar_x, sonar_z, 'k*', markersize=15, label='Sonar Source')
    
    # Mark hydrophone locations (only if they exist and have proper indices)
    if 'hydrophones' in p:
        hydro_x = np.array(p['hydrophones']['x_indices']) * dx
        # Check if z_pos exist, otherwise assume they're on the surface (z=0 or bottom)
        if 'z_pos' in p['hydrophones']:
            hydro_z = np.array(p['hydrophones']['z_pos']) * dz
            if np.ndim(hydro_z) == 0 and hydro_z.size == 1:
                hydro_z = np.full_like(hydro_x, hydro_z)
        else:
            # Assume hydrophones are at the bottom of the domain
            hydro_z = np.zeros_like(hydro_x)
        ax.plot(hydro_x, hydro_z, 'ko', markersize=6, label='Hydrophones')
    else:
        print("No hydrophone data found in parameters.")
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Wave Propagation in 2D Container')
    ax.legend()
    
    # Animation function
    def animate(frame):
        # Skip frames to make animation smoother (every 5th frame)
        actual_frame = min(frame * 5, pressure_history.shape[2] - 1)
        im.set_array(pressure_history[:, :, actual_frame].T)
        ax.set_title(f'Wave Propagation at t = {t[actual_frame]*1000:.1f} ms')
        return [im]
    
    # Create animation
    num_frames = min(3000, pressure_history.shape[2] // 5)  # Limit to 3000 frames max
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                   interval=50, blit=True, repeat=True)
    
    # Save as GIF
    print(f"Creating animation with {num_frames} frames...")
    anim.save(save_filename, writer='pillow', fps=20, dpi=80)
    print(f"Animation saved as {save_filename}")
    
    plt.tight_layout()
    plt.show()
    
    return anim