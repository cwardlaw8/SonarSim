import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def create_3d_wave_animation_new(X, t, p, save_filename='wave_3d_animation.gif', 
                             frame_skip=10, elevation=25, azimuth=-60,
                             z_height_fraction=0.3, cmap='RdBu_r'):
    """
    Create a 3D surface animation of wave propagation with auto-scaling.
    
    Parameters:
    -----------
    X : ndarray
        State matrix (2*Nx*Nz, num_timesteps)
    t : ndarray
        Time vector
    p : dict
        Parameter dictionary containing grid info
    save_filename : str
        Output filename for animation
    frame_skip : int
        Frames to skip between animation frames
    elevation : float
        Viewing angle elevation (degrees)
    azimuth : float
        Viewing angle azimuth (degrees)
    z_height_fraction : float
        Pressure axis will be this fraction of the smaller domain dimension
    cmap : str
        Colormap name
    """
    Nx, Nz = p['Nx'], p['Nz']
    dx, dz = p['dx'], p['dz']
    Lx, Lz = p['Lx'], p['Lz']
    
    # Extract pressure field (second half of state vector)
    N = Nx * Nz
    pressure_history = X[N:2*N, :].reshape(Nx, Nz, -1)
    
    # Auto-scale: make pressure axis proportional to domain
    p_max = np.abs(pressure_history).max() + 1e-12
    target_z_height = z_height_fraction * min(Lx, Lz)
    z_scale = target_z_height / p_max
    
    print(f"Domain: {Lx:.0f}m × {Lz:.0f}m")
    print(f"Pressure range: ±{p_max:.2e} Pa")
    print(f"Z-axis height: {target_z_height:.0f}m (scaled by {z_scale:.2e})")
    
    pressure_scaled = pressure_history * z_scale
    z_range = target_z_height
    
    # Create coordinate grids
    x_coords = np.arange(Nx) * dx
    z_coords = np.arange(Nz) * dz
    X_grid, Z_grid = np.meshgrid(x_coords, z_coords, indexing='ij')
    
    # Set up figure with appropriate size
    aspect = Lx / Lz
    fig_width = 12
    fig_height = max(8, fig_width / aspect * 0.8)
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, projection='3d')
    
    # Source and hydrophone positions
    sonar_x = p['sonar_ix'] * dx
    sonar_z = p['sonar_iz'] * dz
    
    hydro_x, hydro_z = [], []
    if 'hydrophones' in p:
        if 'x_indices' in p['hydrophones']:
            hydro_x = np.array(p['hydrophones']['x_indices']) * dx
            if 'z_indices' in p['hydrophones']:
                hydro_z = np.array(p['hydrophones']['z_indices']) * dz
            elif 'z_pos' in p['hydrophones']:
                hydro_z = np.full_like(hydro_x, p['hydrophones']['z_pos'] * dz)
            else:
                hydro_z = np.zeros_like(hydro_x)
    
    def draw_frame(frame_idx):
        ax.clear()
        
        # Get pressure at this frame
        actual_frame = min(frame_idx * frame_skip, pressure_scaled.shape[2] - 1)
        pressure_frame = pressure_scaled[:, :, actual_frame]
        
        # Plot surface - fully opaque
        surf = ax.plot_surface(X_grid, Z_grid, pressure_frame,
                              cmap=cmap, alpha=1.0,
                              vmin=-z_range, vmax=z_range,
                              linewidth=0, antialiased=True,
                              shade=True)
        
        # Axis labels
        ax.set_xlabel('X (m)', fontsize=10, labelpad=10)
        ax.set_ylabel('Z - Depth (m)', fontsize=10, labelpad=10)
        ax.set_zlabel('Pressure (scaled)', fontsize=10, labelpad=10)
        
        # Title with time
        time_ms = t[actual_frame] * 1000
        ax.set_title(f'Wave Propagation at t = {time_ms:.1f} ms', fontsize=12, pad=20)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Set axis limits
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Lz)
        ax.set_zlim(-z_range, z_range)
        
        # Mark source with vertical line and marker
        ax.plot([sonar_x, sonar_x], [sonar_z, sonar_z], [-z_range, z_range], 
                'k-', linewidth=2)
        ax.scatter([sonar_x], [sonar_z], [z_range], c='yellow', s=100, 
                   marker='*', edgecolors='k', zorder=10)
        
        # Mark hydrophones
        if len(hydro_x) > 0:
            for hx, hz in zip(hydro_x, hydro_z):
                ax.plot([hx, hx], [hz, hz], [-z_range, z_range], 
                        'g-', linewidth=1.5, alpha=0.8)
                ax.scatter([hx], [hz], [-z_range], c='green', s=40, 
                           marker='^', edgecolors='darkgreen', zorder=10)
        
        # Add reference plane at z=0
        ax.plot([0, Lx], [0, 0], [0, 0], 'c-', linewidth=2, label='Surface')
        ax.plot([0, Lx], [Lz, Lz], [0, 0], 'brown', linewidth=2, label='Seafloor')
        
        return [surf]
    
    # Draw initial frame
    draw_frame(0)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-p_max, p_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Pressure (Pa)', fontsize=10)
    
    # Calculate number of frames
    total_frames = pressure_scaled.shape[2]
    num_frames = min(500, total_frames // frame_skip)
    
    print(f"\nCreating animation:")
    print(f"  Total time steps: {total_frames}")
    print(f"  Frame skip: {frame_skip}")
    print(f"  Animation frames: {num_frames}")
    print(f"  Time span: {t[0]*1000:.1f} to {t[-1]*1000:.1f} ms")
    
    # Create animation
    anim = animation.FuncAnimation(fig, draw_frame, frames=num_frames, 
                                   interval=50, repeat=True, blit=False)
    
    # Save
    print(f"\nSaving as {save_filename}...")
    anim.save(save_filename, writer='pillow', fps=15, dpi=100)
    print(f"✓ Saved {save_filename}")
    
    plt.tight_layout()
    plt.show()
    
    return anim


def create_2d_wave_animation_new(X, t, p, save_filename='wave_2d_animation.gif',
                             frame_skip=10, cmap='RdBu_r'):
    """
    Create a 2D heatmap animation - often clearer than 3D.
    
    Parameters:
    -----------
    X : ndarray
        State matrix (2*Nx*Nz, num_timesteps)
    t : ndarray
        Time vector
    p : dict
        Parameter dictionary
    save_filename : str
        Output filename
    frame_skip : int
        Frames to skip
    cmap : str
        Colormap
    """
    Nx, Nz = p['Nx'], p['Nz']
    dx, dz = p['dx'], p['dz']
    Lx, Lz = p['Lx'], p['Lz']
    
    # Extract pressure
    N = Nx * Nz
    pressure_history = X[N:2*N, :].reshape(Nx, Nz, -1)
    
    # Global color scale
    vmax = np.abs(pressure_history).max()
    
    # Figure setup
    aspect = Lx / Lz
    fig_width = 12
    fig_height = max(4, fig_width / aspect * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Initial plot
    im = ax.imshow(pressure_history[:, :, 0].T, 
                   extent=[0, Lx, Lz, 0],
                   cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect='auto', origin='upper')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pressure (Pa)')
    
    # Mark source
    sonar_x = p['sonar_ix'] * dx
    sonar_z = p['sonar_iz'] * dz
    source_marker, = ax.plot(sonar_x, sonar_z, 'y*', markersize=15, 
                              markeredgecolor='k', markeredgewidth=1)
    
    # Mark hydrophones
    if 'hydrophones' in p:
        if 'x_indices' in p['hydrophones']:
            hydro_x = np.array(p['hydrophones']['x_indices']) * dx
            if 'z_pos' in p['hydrophones']:
                hydro_z = np.full_like(hydro_x, p['hydrophones']['z_pos'] * dz)
            else:
                hydro_z = np.zeros_like(hydro_x)
            ax.plot(hydro_x, hydro_z, 'ws', markersize=6, 
                    markeredgecolor='k', linewidth=1.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    title = ax.set_title(f't = {t[0]*1000:.1f} ms')
    
    # Boundary labels
    ax.axhline(0, color='cyan', linewidth=2, linestyle='-')
    ax.axhline(Lz, color='brown', linewidth=2, linestyle='-')
    
    def animate(frame):
        actual_frame = min(frame * frame_skip, pressure_history.shape[2] - 1)
        im.set_array(pressure_history[:, :, actual_frame].T)
        title.set_text(f't = {t[actual_frame]*1000:.1f} ms')
        return [im, title]
    
    total_frames = pressure_history.shape[2]
    num_frames = min(500, total_frames // frame_skip)
    
    print(f"Creating 2D animation: {num_frames} frames")
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                   interval=50, blit=True, repeat=True)
    
    print(f"Saving as {save_filename}...")
    anim.save(save_filename, writer='pillow', fps=15, dpi=100)
    print(f"✓ Saved {save_filename}")
    
    plt.tight_layout()
    plt.show()
    
    return anim



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
    pressure_history = X[N:2*N, :].reshape(Nx, Nz, -1)
    
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
    
    # Animation function
    def animate(frame):
        ax.clear()
        
        actual_frame = min(frame * frame_skip, pressure_history.shape[2] - 1)
        
        surf = ax.plot_surface(X_grid, Z_grid, pressure_history[:, :, actual_frame],
                              cmap='RdBu_r', alpha=0.8,
                              vmin=-z_range, vmax=z_range)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('Pressure (Pa)')
        ax.set_title(f'3D Wave Propagation at t = {t[actual_frame]*1000:.1f} ms')
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlim(0, Nx*dx)
        ax.set_ylim(0, Nz*dz)
        ax.set_zlim(-z_range, z_range)
        
        ax.plot([sonar_x, sonar_x], [sonar_z, sonar_z], [-z_range, z_range], 
                'k-', linewidth=3)
        
        return [surf]
    
    total_frames = pressure_history.shape[2]
    num_frames = min(300, total_frames // frame_skip)
    
    print(f"Creating 3D animation with:")
    print(f"  - Total time steps: {total_frames}")
    print(f"  - Frame skip: {frame_skip}")
    print(f"  - Animation frames: {num_frames}")
    print(f"  - Time span: {t[0]*1000:.1f} to {t[-1]*1000:.1f} ms")
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                   interval=100, repeat=True)
    
    print(f"Saving 3D animation as {save_filename}...")
    anim.save(save_filename, writer='pillow', fps=10, dpi=80)
    print(f"3D animation saved as {save_filename}")
    
    plt.tight_layout()
    plt.show()
    
    return anim