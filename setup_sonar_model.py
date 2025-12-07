"""
Wrapper function for consistent sonar model setup with eval_u_Sonar_20.
"""
import numpy as np
from scipy import sparse as sp
from getParam_Sonar import getParam_Sonar
from eval_u_Sonar import eval_u_Sonar_20_const
from eval_f_Sonar import eval_f_Sonar
from eval_g_Sonar import eval_g_Sonar


def setup_sonar_model_full(Nx=301, Nz=51, Lx=5625, Lz=937.5, f0=20, 
                      BC=True, UseSparseMatrices=True,
                      source_position='surface', hydrophone_config='Horizontal',
                      t_extra=0):
    """
    Complete sonar model setup with eval_u_Sonar_20_const (constant 20Hz signal).
    
    Parameters:
    -----------
    Nx, Nz : int
        Grid dimensions
    Lx, Lz : float
        Domain size in meters
    f0 : float
        Signal frequency in Hz (for wavelength calculations)
    BC : bool
        Use boundary conditions (True recommended)
    UseSparseMatrices : bool
        Use sparse matrices (True recommended for large grids)
    source_position : str or tuple
        'center': center of domain, colocated with center hydrophone (mid-depth)
        'surface': at surface (z=0), center of domain horizontally
        (ix, iz): explicit grid indices
    hydrophone_config : str or dict
        'horizontal': 5 equally-spaced hydrophones at mid-depth
        'vertical': 5 equally-spaced hydrophones at center x-position
        dict: custom configuration with 'x_indices' and 'z_pos' or 'z_indices'
    t_extra : float
        Additional simulation time beyond default (seconds)
    
    Returns:
    --------
    model : dict
        Complete model configuration containing:
        - p : parameter dict (from getParam_Sonar)
        - x_start : initial state
        - t_start, t_stop : time span
        - max_dt_FE : maximum stable timestep for explicit methods
        - eval_f : force function (eval_f_Sonar)
        - eval_u : input function (eval_u_Sonar_20)
        - eval_u_scaled : scaled input function (accounts for cell area)
        - eval_g : output function (eval_g_Sonar)
        - Nx, Nz, Lx, Lz : grid parameters
        - f0 : signal frequency
        - wavelength : acoustic wavelength
        - ppw : points per wavelength
    """
    
    # Get base parameters from getParam_Sonar
    p, x_start, t_start, t_stop, max_dt_FE = getParam_Sonar(
        Nx, Nz, Lx, Lz, UseSparseMatrices=UseSparseMatrices, BC=BC
    )
    
    # Extend simulation time
    t_stop += t_extra
    
    # Configure source position
    if source_position == 'center':
        # Center of domain, mid-depth (colocated with center hydrophone)
        p['sonar_ix'] = Nx // 2
        p['sonar_iz'] = Nz // 2
    elif source_position == 'surface':
        # Near-surface (z=1, just below surface), center of domain horizontally
        # NOTE: Can't use z=0 because that's the pressure-release boundary (p=0)
        p['sonar_ix'] = Nx // 2
        p['sonar_iz'] = 1
    elif isinstance(source_position, (tuple, list)) and len(source_position) == 2:
        p['sonar_ix'] = source_position[0]
        p['sonar_iz'] = source_position[1]
    else:
        raise ValueError(f"Unknown source_position: {source_position}")
    
    # Rebuild B matrix with new source location
    N = Nx * Nz
    source_idx = p['sonar_ix'] * Nz + p['sonar_iz']
    
    B_lil = sp.lil_matrix((2*N, 1), dtype=float)
    B_lil[source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
    p['B'] = B_lil.tocsr()
    
    # Configure hydrophones
    if hydrophone_config == 'horizontal':
        # 5 equally-spaced hydrophones at mid-depth, avoiding edges (absorbing boundaries)
        z_pos = Nz // 2
        n_phones = 5
        # Divide domain into n_phones+1 segments, place hydrophones at segment boundaries (excluding edges)
        x_indices = [(Nx - 1) * (i + 1) // (n_phones + 1) for i in range(n_phones)]
        p['hydrophones'] = {
            'z_pos': z_pos,
            'x_indices': x_indices,
            'n_phones': n_phones
        }
    elif hydrophone_config == 'vertical':
        # 5 equally-spaced hydrophones vertically at center x-position, avoiding edges
        x_pos = Nx // 2
        n_phones = 5
        # Divide domain into n_phones+1 segments, place hydrophones at segment boundaries (excluding edges)
        z_indices = [(Nz - 1) * (i + 1) // (n_phones + 1) for i in range(n_phones)]
        p['hydrophones'] = {
            'x_pos': x_pos,
            'z_indices': z_indices,
            'n_phones': n_phones
        }
    elif isinstance(hydrophone_config, dict):
        # Custom configuration
        p['hydrophones'] = hydrophone_config
    else:
        raise ValueError(f"Unknown hydrophone_config: {hydrophone_config}")
    
    # Wavelength calculations
    wavelength = p['c'] / f0
    ppw = wavelength / p['dx']
    
    # Create scaled input function (accounts for cell area)
    def eval_u_scaled(t):
        return (p['dx'] * p['dz']) * eval_u_Sonar_20_const(t)
    
    # Assemble complete model dictionary
    model = {
        # Core parameters
        'p': p,
        'x_start': x_start,
        't_start': t_start,
        't_stop': t_stop,
        'max_dt_FE': max_dt_FE,
        
        # Functions
        'eval_f': eval_f_Sonar,
        'eval_u': eval_u_Sonar_20_const,
        'eval_u_scaled': eval_u_scaled,
        'eval_g': eval_g_Sonar,
        
        # Grid info
        'Nx': Nx,
        'Nz': Nz,
        'Lx': Lx,
        'Lz': Lz,
        'dx': p['dx'],
        'dz': p['dz'],
        
        # Acoustic properties
        'f0': f0,
        'wavelength': wavelength,
        'ppw': ppw,
        'c': p['c'],
        
        # Source/receiver info
        'source_ix': p['sonar_ix'],
        'source_iz': p['sonar_iz'],
        'hydrophones': p['hydrophones']
    }
    
    return model

def setup_sonar_model(Nx=301, Nz=21, Lx=8000, Lz=1000, f0=20, 
                      BC=True, UseSparseMatrices=True,
                      source_position='vertical', hydrophone_config='horizontal',
                      t_extra=0.0):
    """
    Wrapper for setup_sonar_model_full using smaller grid for quicker runs.
    """
    return setup_sonar_model_full(Nx=Nx, Nz=Nz, Lx=Lx, Lz=Lz, f0=f0,
                                  BC=BC, UseSparseMatrices=UseSparseMatrices,
                                  hydrophone_config=hydrophone_config,
                                  source_position=source_position,
                                  t_extra=t_extra)


def print_model_info(model, verbose=True):
    """Print summary of model configuration."""
    
    print("="*70)
    print("SONAR MODEL CONFIGURATION")
    print("="*70)
    
    # Grid information
    print(f"\nGrid: {model['Nx']} × {model['Nz']} = {model['Nx']*model['Nz']:,} cells")
    print(f"Domain: {model['Lx']:.0f}m × {model['Lz']:.0f}m")
    print(f"Spacing: dx = {model['dx']:.4f}m, dz = {model['dz']:.4f}m")
    
    # Acoustic properties
    print(f"\nAcoustic Properties:")
    print(f"  Sound speed: c = {model['c']} m/s")
    print(f"  Frequency: f₀ = {model['f0']} Hz")
    print(f"  Wavelength: λ = {model['wavelength']:.2f}m")
    print(f"  Resolution: {model['ppw']:.1f} points per wavelength")
    print(f"  Domain coverage: {model['Lx']/model['wavelength']:.1f}λ × {model['Lz']/model['wavelength']:.1f}λ")
    
    # Source information
    source_x = model['source_ix'] * model['dx']
    source_z = model['source_iz'] * model['dz']
    print(f"\nSource Position:")
    print(f"  Grid indices: ({model['source_ix']}, {model['source_iz']})")
    print(f"  Physical: x = {source_x:.1f}m, z = {source_z:.1f}m")
    
    # Hydrophone information
    hydro = model['hydrophones']
    print(f"\nHydrophones: {hydro['n_phones']} receivers")
    if 'x_indices' in hydro:
        print(f"  Type: Horizontal array at z = {hydro['z_pos'] * model['dz']:.1f}m")
        for i, x_idx in enumerate(hydro['x_indices']):
            x_pos = x_idx * model['dx']
            print(f"    H{i+1}: x = {x_pos:.1f}m")
    elif 'z_indices' in hydro:
        print(f"  Type: Vertical array at x = {hydro['x_pos'] * model['dx']:.1f}m")
        print(f"    Depths: z = {min(hydro['z_indices'])*model['dz']:.1f}m to {max(hydro['z_indices'])*model['dz']:.1f}m")
    
    # Time integration
    print(f"\nTime Integration:")
    print(f"  Time span: {model['t_start']:.3f}s to {model['t_stop']:.3f}s ({model['t_stop']*1000:.1f}ms)")
    print(f"  Max stable dt (CFL): {model['max_dt_FE']*1e6:.2f} μs")
    
    # State vector
    N_states = model['x_start'].shape[0]
    print(f"\nState Vector: {N_states:,} DOFs")
    
    if verbose:
        # Additional diagnostics
        print(f"\nRecommended Settings:")
        nyquist_dt = 1 / (2 * model['f0'])
        if model['max_dt_FE'] <= nyquist_dt:
            print(f"  ✓ CFL timestep satisfies Nyquist (limit = {nyquist_dt*1e6:.2f} μs)")
        else:
            print(f"  ✗ WARNING: CFL > Nyquist! Limit = {nyquist_dt*1e6:.2f} μs")
        
        if model['ppw'] >= 4:
            print(f"  ✓ Spatial resolution adequate ({model['ppw']:.1f} ppw ≥ 4)")
        else:
            print(f"  ✗ WARNING: Low spatial resolution ({model['ppw']:.1f} ppw < 4)")
    
    print("="*70)


# Example usage
if __name__ == "__main__":
    # Default setup with horizontal array
    model = setup_sonar_model()
    print_model_info(model)
