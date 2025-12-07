import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def create_3d_wave_animation(X, t, p, save_filename='wave_3d_animation.gif', 
                           frame_skip=10, elevation=30, azimuth=45):
    """
    Create a 3D surface animation of wave propagation
    
    Parameters:
    X: state matrix (2*Nx*Nz, num_timesteps)
    t: time vector
    p: parameter dictionary containing grid info
    save_filename: name for the saved animation
    frame_skip: how many frames to skip between animation frames
    elevation: viewing angle elevation (degrees)
    azimuth: viewing angle azimuth (degrees)
    """
    Nx, Nz = p['Nx'], p['Nz']
    dx, dz = p['dx'], p['dz']
    
    # Extract pressure field (second half of state vector)
    N = Nx * Nz
    pressure_history = X[N:2*N, :].reshape(Nx, Nz, -1)  # CHANGED: was X[:N, :]
    
    # Create coordinate grids
    x_coords = np.arange(Nx) * dx
    z_coords = np.arange(Nz) * dz
    X_grid, Z_grid = np.meshgrid(x_coords, z_coords, indexing='ij')
    
    # Set up the 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find global min/max for consistent z-scaling
    pressure_min = np.min(pressure_history)
    pressure_max = np.max(pressure_history)
    z_range = max(abs(pressure_min), abs(pressure_max))
    
    # Create initial 3D surface
    surf = ax.plot_surface(X_grid, Z_grid, pressure_history[:, :, 0],
                          cmap='RdBu_r', alpha=0.8,
                          vmin=-z_range, vmax=z_range)
    
    # Set up the plot
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Pressure (Pa)')
    ax.set_title('3D Wave Propagation Animation')
    
    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Set consistent axis limits
    ax.set_xlim(0, Nx*dx)
    ax.set_ylim(0, Nz*dz)
    ax.set_zlim(-z_range, z_range)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Mark sonar source location with a vertical line
    sonar_x = p['sonar_ix'] * dx
    sonar_z = p['sonar_iz'] * dz
    ax.plot([sonar_x, sonar_x], [sonar_z, sonar_z], [-z_range, z_range], 
            'k-', linewidth=3, label='Sonar Source')
    
    # Mark hydrophone locations if they exist
    if 'hydrophones' in p:
        hydro_x = np.array(p['hydrophones']['x_indices']) * dx
        if 'z_indices' in p['hydrophones']:
            hydro_z = np.array(p['hydrophones']['z_indices']) * dz
        else:
            hydro_z = np.zeros_like(hydro_x)
        
        # Plot hydrophones as vertical lines
        for hx, hz in zip(hydro_x, hydro_z):
            ax.plot([hx, hx], [hz, hz], [-z_range, z_range], 
                    'g-', linewidth=2, alpha=0.7)
    
    # Animation function
    def animate(frame):
        ax.clear()  # Clear the previous surface
        
        # Calculate actual frame index
        actual_frame = min(frame * frame_skip, pressure_history.shape[2] - 1)
        
        # Create new surface
        surf = ax.plot_surface(X_grid, Z_grid, pressure_history[:, :, actual_frame],
                              cmap='RdBu_r', alpha=0.8,
                              vmin=-z_range, vmax=z_range)
        
        # Reset plot properties (since we cleared)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('Pressure (Pa)')
        ax.set_title(f'3D Wave Propagation at t = {t[actual_frame]*1000:.1f} ms')
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlim(0, Nx*dx)
        ax.set_ylim(0, Nz*dz)
        ax.set_zlim(-z_range, z_range)
        
        # Re-add markers
        ax.plot([sonar_x, sonar_x], [sonar_z, sonar_z], [-z_range, z_range], 
                'k-', linewidth=3)
        
        if 'hydrophones' in p:
            for hx, hz in zip(hydro_x, hydro_z):
                ax.plot([hx, hx], [hz, hz], [-z_range, z_range], 
                        'g-', linewidth=2, alpha=0.7)
        
        return [surf]
    
    # Calculate number of frames
    total_frames = pressure_history.shape[2]
    num_frames = min(3000, total_frames // frame_skip)  # Limit to 300 frames max
    
    print(f"Creating 3D animation with:")
    print(f"  - Total time steps: {total_frames}")
    print(f"  - Frame skip: {frame_skip}")
    print(f"  - Animation frames: {num_frames}")
    print(f"  - Time span: {t[0]*1000:.1f} to {t[-1]*1000:.1f} ms")
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                   interval=100, repeat=True)
    
    # Save as GIF
    print(f"Saving 3D animation as {save_filename}...")
    anim.save(save_filename, writer='pillow', fps=10, dpi=80)
    print(f"3D animation saved as {save_filename}")
    
    plt.tight_layout()
    plt.show()
    
    return anim