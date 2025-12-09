import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch

def visualize_sonar_setup(p, show_grid=True, show_boundary_labels=True):
    """
    Visualize the sonar simulation domain showing:
    - Grid points (optional)
    - Source location
    - Hydrophone array
    - Boundary conditions
    - Domain dimensions
    
    Parameters:
    -----------
    p : dict
        Parameter dictionary from getParam_Sonar
    show_grid : bool
        Whether to show all grid points
    show_boundary_labels : bool
        Whether to label the boundary conditions
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create grid coordinates
    x = np.linspace(0, p['Lx'], p['Nx'])
    z = np.linspace(0, p['Lz'], p['Nz'])
    margin_x = p['Lx'] * 0.05
    margin_z = p['Lz'] * 0.05
    
    # Plot grid if requested
    if show_grid:
        # Plot grid points as small dots
        X, Z = np.meshgrid(x, z)
        ax.plot(X.T, Z.T, 'k-', alpha=0.1, linewidth=0.5)
        ax.plot(X, Z, 'k-', alpha=0.1, linewidth=0.5)
        ax.plot(X.flatten(), Z.flatten(), 'k.', markersize=1, alpha=0.3)
    
    # Plot boundary conditions with colored edges
    # Top boundary (pressure-release, sea surface)
    ax.plot([0, p['Lx']], [0, 0], 'c-', linewidth=3, label='Pressure-release (sea surface)')
#     ax.plot([0, p['Lx']], [0, 0], 'blue', linewidth=3, label='Pressure-release (sea surface)')
    
    # Bottom boundary (rigid, seafloor)
    ax.plot([0, p['Lx']], [p['Lz'], p['Lz']], 'brown', linewidth=3, label='Rigid (seafloor)')
    
    # Left boundary (absorbing)
    ax.plot([0, 0], [0, p['Lz']], 'g--', linewidth=3, label='Absorbing boundary')
    
    # Right boundary (absorbing)
    ax.plot([p['Lx'], p['Lx']], [0, p['Lz']], 'g--', linewidth=3)
    
    # Add boundary labels if requested
    if show_boundary_labels:
        label_kwargs = dict(textcoords='offset points', clip_on=False, fontsize=10, weight='bold')
        # Top
        ax.annotate('Pressure-release (Sea Surface)',
                    xy=(p['Lx']/2, 0), xytext=(0, -10),
                #     ha='center', va='top', color='cyan', **label_kwargs)
                    ha='center', va='top', color='black', **label_kwargs)
        # Bottom
        ax.annotate('Rigid (Seafloor)',
                    xy=(p['Lx']/2, p['Lz']), xytext=(0, 10),
                    ha='center', va='bottom', color='brown', **label_kwargs)
        # Left
        ax.annotate('Absorbing',
                    xy=(0, p['Lz']/2), xytext=(-10, 0),
                    ha='right', va='center', color='green', rotation=90, **label_kwargs)
        # Right
        ax.annotate('Absorbing',
                    xy=(p['Lx'], p['Lz']/2), xytext=(10, 0),
                    ha='left', va='center', color='green', rotation=90, **label_kwargs)
    
    # Plot source location
    source_x = p['sonar_ix'] * p['dx']
    source_z = p['sonar_iz'] * p['dz']
    ax.plot(source_x, source_z, 'r*', markersize=20, label='Sonar source', 
            markeredgecolor='darkred', markeredgewidth=1)
    
    # Add a circle around the source to make it more visible
    circle = plt.Circle((source_x, source_z), p['dx']*2, 
                        color='red', fill=False, linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    
    # Plot hydrophone array (handle both horizontal and vertical configurations)
    hydro = p['hydrophones']
    if 'z_pos' in hydro and 'x_indices' in hydro:
        # Horizontal array: all hydrophones at same depth
        hydrophone_z = hydro['z_pos'] * p['dz']
        for i, x_idx in enumerate(hydro['x_indices']):
            hydrophone_x = x_idx * p['dx']
            ax.plot(hydrophone_x, hydrophone_z, 'b^', markersize=12, 
                    markeredgecolor='darkblue', markeredgewidth=1)
            # Label each hydrophone
            ax.text(hydrophone_x, hydrophone_z - p['dz']*1.5, f'H{i+1}', 
                    ha='center', va='top', fontsize=8, color='blue')
    elif 'x_pos' in hydro and 'z_indices' in hydro:
        # Vertical array: all hydrophones at same x position
        hydrophone_x = hydro['x_pos'] * p['dx']
        for i, z_idx in enumerate(hydro['z_indices']):
            hydrophone_z = z_idx * p['dz']
            ax.plot(hydrophone_x, hydrophone_z, 'b^', markersize=12, 
                    markeredgecolor='darkblue', markeredgewidth=1)
            # Label each hydrophone
            ax.text(hydrophone_x + p['dx']*1.5, hydrophone_z, f'H{i+1}', 
                    ha='left', va='center', fontsize=8, color='blue')
    elif 'x_indices' in hydro and 'z_indices' in hydro:
        # Custom paired coordinates
        for i, (x_idx, z_idx) in enumerate(zip(hydro['x_indices'], hydro['z_indices'])):
            hydrophone_x = x_idx * p['dx']
            hydrophone_z = z_idx * p['dz']
            ax.plot(hydrophone_x, hydrophone_z, 'b^', markersize=12,
                    markeredgecolor='darkblue', markeredgewidth=1)
            ax.text(hydrophone_x + p['dx']*0.8, hydrophone_z - p['dz']*0.8, f'H{i+1}',
                    ha='left', va='center', fontsize=8, color='blue')
    
    # Add hydrophone array label
    if p['hydrophones']['n_phones'] > 0:
        ax.plot([], [], 'b^', markersize=12, label='Hydrophone array')
    
    # Add wavelength indicator for scale
    wavelength = p['c'] / 20  # For 20 Hz source
    wave_y = p['Lz'] * 0.8  # keep clear of bottom info box
    wave_x0 = p['Lx'] * 0.99
    wave_x1 = wave_x0 + wavelength
    # keep the scale bar away from the right edge if λ is large
    if wave_x1 > p['Lx'] * 0.9:
        shift = wave_x1 - p['Lx'] * 0.9
        wave_x0 -= shift
        wave_x1 -= shift
    ax.plot([wave_x0, wave_x1], [wave_y, wave_y], 'k-', linewidth=2)
    ax.annotate(f'λ @ 20Hz = {wavelength:.1f}m',
                xy=((wave_x0 + wave_x1) / 2, wave_y),
                xytext=(0, -8), textcoords='offset points',
                ha='center', va='top', fontsize=9, clip_on=False)
    
    # Grid information box
    info_text = (
        f"Spacing: Δx={p['dx']:.2f}m, Δz={p['dz']:.2f}m"
    )
    
    # Add text box with parameters
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.6, 0.1, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax.set_xlabel('X Distance (m)', fontsize=12, weight='bold')
    ax.set_ylabel('Z Depth (m)', fontsize=12, weight='bold')
    ax.set_title('Sonar Simulation Domain Setup', fontsize=14, weight='bold')
    
    # Invert y-axis so depth increases downward (ocean convention)
    ax.invert_yaxis()
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend
    #ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits with small margin
    ax.set_xlim(-margin_x, p['Lx'] + margin_x)
    ax.set_ylim(p['Lz'] + margin_z, -margin_z)
    
    plt.tight_layout()
    return fig, ax
