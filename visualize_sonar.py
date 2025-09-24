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
    
    # Bottom boundary (rigid, seafloor)
    ax.plot([0, p['Lx']], [p['Lz'], p['Lz']], 'brown', linewidth=3, label='Rigid (seafloor)')
    
    # Left boundary (absorbing)
    ax.plot([0, 0], [0, p['Lz']], 'g--', linewidth=3, label='Absorbing boundary')
    
    # Right boundary (absorbing)
    ax.plot([p['Lx'], p['Lx']], [0, p['Lz']], 'g--', linewidth=3)
    
    # Add boundary labels if requested
    if show_boundary_labels:
        # Top
        ax.text(p['Lx']/2, -p['Lz']*0.02, 'Pressure-release (Sea Surface)', 
                ha='center', va='top', fontsize=10, color='cyan', weight='bold')
        # Bottom
        ax.text(p['Lx']/2, p['Lz']*1.02, 'Rigid (Seafloor)', 
                ha='center', va='bottom', fontsize=10, color='brown', weight='bold')
        # Left
        ax.text(-p['Lx']*0.02, p['Lz']/2, 'Absorbing', 
                ha='right', va='center', fontsize=10, color='green', 
                rotation=90, weight='bold')
        # Right
        ax.text(p['Lx']*1.02, p['Lz']/2, 'Absorbing', 
                ha='left', va='center', fontsize=10, color='green', 
                rotation=90, weight='bold')
    
    # Plot source location
    source_x = p['sonar_ix'] * p['dx']
    source_z = p['sonar_iz'] * p['dz']
    ax.plot(source_x, source_z, 'r*', markersize=20, label='Sonar source', 
            markeredgecolor='darkred', markeredgewidth=1)
    
    # Add a circle around the source to make it more visible
    circle = plt.Circle((source_x, source_z), p['dx']*2, 
                        color='red', fill=False, linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    
    # Plot hydrophone array
    hydrophone_z = p['hydrophones']['z_pos'] * p['dz']
    for i, x_idx in enumerate(p['hydrophones']['x_indices']):
        hydrophone_x = x_idx * p['dx']
        ax.plot(hydrophone_x, hydrophone_z, 'b^', markersize=12, 
                markeredgecolor='darkblue', markeredgewidth=1)
        # Label each hydrophone
        ax.text(hydrophone_x, hydrophone_z - p['dz']*1.5, f'H{i}', 
                ha='center', va='top', fontsize=8, color='blue')
    
    # Add hydrophone array label
    if p['hydrophones']['n_phones'] > 0:
        first_hydro_x = p['hydrophones']['x_indices'][0] * p['dx']
        ax.plot([], [], 'b^', markersize=12, label='Hydrophone array')
    
    # Add wavelength indicator for scale
    wavelength = p['c'] / 20  # For 20 Hz source
    ax.plot([p['Lx']*0.7, p['Lx']*0.7 + wavelength], 
            [p['Lz']*0.9, p['Lz']*0.9], 'k-', linewidth=2)
    ax.text(p['Lx']*0.7 + wavelength/2, p['Lz']*0.92, 
            f'λ @ 20Hz = {wavelength:.1f}m', ha='center', fontsize=9)
    
    # Grid information box
    info_text = (
        f"Spacing: Δx={p['dx']:.2f}m, Δz={p['dz']:.2f}m"
    )
    
    # Add text box with parameters
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.6, 0.1, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax.set_xlabel('X Distance (m)', fontsize=12)
    ax.set_ylabel('Z Depth (m)', fontsize=12)
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
    margin_x = p['Lx'] * 0.05
    margin_z = p['Lz'] * 0.05
    ax.set_xlim(-margin_x, p['Lx'] + margin_x)
    ax.set_ylim(p['Lz'] + margin_z, -margin_z)
    
    plt.tight_layout()
    return fig, ax

