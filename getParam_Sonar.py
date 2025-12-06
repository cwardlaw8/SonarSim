import numpy as np
import scipy.sparse as sp

def getParam_Sonar(Nx, Nz, Lx, Lz, UseSparseMatrices=True, absorb_strength=5.0, 
                   alpha=0.001, enforce_surface_BC=True):
    """
    Defines the parameters for 2D acoustic wave equation for sonar propagation.
    Returns matrices for the linear system representation dx/dt = p.A x + p.B u
    where state x = [w_1, ..., w_N, p_1, ..., p_N]^T (velocity and pressure)

    INPUT:
    Nx                  number of grid points in x direction
    Nz                  number of grid points in z direction
    Lx                  total length in x direction (m)
    Lz                  total length in z direction (m)
    UseSparseMatrices   use sparse matrix format (default: True)
    absorb_strength     strength of absorbing boundaries (default: 5.0)
    alpha               global absorption coefficient (1/s) (default: 0.001)
    enforce_surface_BC  enforce p=0 at surface z=0 (default: True)

    OUTPUTS:
    p         dictionary containing system matrices and parameters
    x_start   initial state vector [w; p]
    t_start   initial time (0)
    t_stop    simulation end time (acoustic crossing time)
    max_dt_FE maximum stable timestep for Forward Euler
    """
    
    p = {
        'c': 1500.0,           # (m/s) speed of sound in water
        'rho': 1025,           # (kg/m³) density of seawater
        'alpha': alpha,        # (1/s) global absorption coefficient
        'Nx': Nx,              # grid points in x
        'Nz': Nz,              # grid points in z
        'Lx': Lx,              # domain size x (m)
        'Lz': Lz,              # domain size z (m)
        'sonar_ix': Nx//2,     # source position x index
        'sonar_iz': 1,         # source position z index (near surface)
        'absorb_strength': absorb_strength
    }
    
    # Hydrophone array configuration
    n_phones = 5
    p['hydrophones'] = {
        'z_pos': Nz // 2,
        'x_indices': np.linspace(Nx//8, Nx - Nx//8, n_phones, dtype=int).tolist(),
        'n_phones': n_phones
    }

    p['dx'] = Lx / (Nx - 1)
    p['dz'] = Lz / (Nz - 1)
    
    N = Nx * Nz
    idx = lambda i, j: i * Nz + j  # Flattened index
    
    c2_dx2 = (p['c']**2) / (p['dx']**2)
    c2_dz2 = (p['c']**2) / (p['dz']**2)
    
    # =========================================================================
    # Build Laplacian L using efficient diagonal construction
    # =========================================================================
    # L represents ∇²p (Laplacian operator) with boundary conditions:
    #   - Top (z=0): pressure release (p=0) via ghost point p_{i,-1} = -p_{i,0}
    #   - Bottom (z=Lz): rigid wall (∂p/∂n=0) via ghost point p_{i,Nz} = p_{i,Nz-1}
    #   - Left/Right (x=0, x=Lx): one-sided differences (absorbing approximation)
    
    if UseSparseMatrices:
        # Use LIL format for efficient row-by-row construction
        L = sp.lil_matrix((N, N))
        
        for i in range(Nx):
            for j in range(Nz):
                k = idx(i, j)
                
                # Determine boundary type
                is_interior_x = (1 <= i < Nx - 1)
                is_interior_z = (1 <= j < Nz - 1)
                
                # Main diagonal (same for all points)
                L[k, k] = -2 * (c2_dx2 + c2_dz2)
                
                # X-direction connections
                if is_interior_x:
                    # Interior in x: standard central difference
                    L[k, idx(i-1, j)] = c2_dx2
                    L[k, idx(i+1, j)] = c2_dx2
                elif i == 0:
                    # Left boundary: one-sided (forward)
                    L[k, idx(i+1, j)] = 2 * c2_dx2
                else:  # i == Nx-1
                    # Right boundary: one-sided (backward)
                    L[k, idx(i-1, j)] = 2 * c2_dx2
                
                # Z-direction connections
                if is_interior_z:
                    # Interior in z: standard central difference
                    L[k, idx(i, j-1)] = c2_dz2
                    L[k, idx(i, j+1)] = c2_dz2
                elif j == 0:
                    # Top boundary (pressure release): one-sided (forward)
                    L[k, idx(i, j+1)] = 2 * c2_dz2
                else:  # j == Nz-1
                    # Bottom boundary (rigid): one-sided (backward)
                    L[k, idx(i, j-1)] = 2 * c2_dz2
        
        # Convert to CSR for efficient matrix-vector products
        L = L.tocsr()
        
    else:
        # Dense matrix construction (for debugging/small grids only)
        L = np.zeros((N, N))
        
        for i in range(Nx):
            for j in range(Nz):
                k = idx(i, j)
                
                # Interior points: standard 5-point stencil
                if 1 <= i < Nx - 1 and 1 <= j < Nz - 1:
                    L[k, k] = -2 * (c2_dx2 + c2_dz2)
                    L[k, idx(i-1, j)] = c2_dx2
                    L[k, idx(i+1, j)] = c2_dx2
                    L[k, idx(i, j-1)] = c2_dz2
                    L[k, idx(i, j+1)] = c2_dz2
                
                # Boundary points (one-sided differences)
                elif i == 0 and 1 <= j < Nz - 1:  # Left
                    L[k, k] = -2*c2_dx2 - 2*c2_dz2
                    L[k, idx(i+1, j)] = 2*c2_dx2
                    L[k, idx(i, j-1)] = c2_dz2
                    L[k, idx(i, j+1)] = c2_dz2
                
                elif i == Nx - 1 and 1 <= j < Nz - 1:  # Right
                    L[k, k] = -2*c2_dx2 - 2*c2_dz2
                    L[k, idx(i-1, j)] = 2*c2_dx2
                    L[k, idx(i, j-1)] = c2_dz2
                    L[k, idx(i, j+1)] = c2_dz2
                
                elif j == 0 and 1 <= i < Nx - 1:  # Top (pressure release)
                    L[k, k] = -2*c2_dx2 - 2*c2_dz2
                    L[k, idx(i-1, j)] = c2_dx2
                    L[k, idx(i+1, j)] = c2_dx2
                    L[k, idx(i, j+1)] = 2*c2_dz2
                
                elif j == Nz - 1 and 1 <= i < Nx - 1:  # Bottom (rigid)
                    L[k, k] = -2*c2_dx2 - 2*c2_dz2
                    L[k, idx(i-1, j)] = c2_dx2
                    L[k, idx(i+1, j)] = c2_dx2
                    L[k, idx(i, j-1)] = 2*c2_dz2
                
                # Corners
                elif i == 0 and j == 0:
                    L[k, k] = -2*c2_dx2 - 2*c2_dz2
                    L[k, idx(i+1, j)] = 2*c2_dx2
                    L[k, idx(i, j+1)] = 2*c2_dz2
                
                elif i == Nx-1 and j == 0:
                    L[k, k] = -2*c2_dx2 - 2*c2_dz2
                    L[k, idx(i-1, j)] = 2*c2_dx2
                    L[k, idx(i, j+1)] = 2*c2_dz2
                
                elif i == 0 and j == Nz-1:
                    L[k, k] = -2*c2_dx2 - 2*c2_dz2
                    L[k, idx(i+1, j)] = 2*c2_dx2
                    L[k, idx(i, j-1)] = 2*c2_dz2
                
                elif i == Nx-1 and j == Nz-1:
                    L[k, k] = -2*c2_dx2 - 2*c2_dz2
                    L[k, idx(i-1, j)] = 2*c2_dx2
                    L[k, idx(i, j-1)] = 2*c2_dz2
    
    # =========================================================================
    # Spatially-varying damping for absorbing boundaries
    # =========================================================================
    # Damping enters the velocity equation: dw/dt = -damping*w + L*p
    # Add extra damping near left/right boundaries to reduce reflections
    
    absorb_damping_field = np.zeros(N)
    absorb_coeff = absorb_strength * max(c2_dx2, c2_dz2)
    
    # Vectorized boundary assignment (much faster than loops)
    absorb_damping_field[0:Nz] = absorb_coeff              # Left boundary (i=0)
    absorb_damping_field[(Nx-1)*Nz:N] = absorb_coeff      # Right boundary (i=Nx-1)
    
    # Total damping: global + absorbing
    total_damping = p['alpha'] + absorb_damping_field
    
    # Store for diagnostics
    p['absorb_damping_field'] = absorb_damping_field
    p['total_damping'] = total_damping
    
    # =========================================================================
    # Build state-space system: dx/dt = Ax + Bu
    # =========================================================================
    # State ordering: x = [w, p] where w = dp/dt
    # Equations:
    #   dw/dt = -total_damping*w + L*p + source
    #   dp/dt = w
    # 
    # Matrix form:
    #   A = [-diag(total_damping),  L ]
    #       [ I,                    0 ]
    
    if UseSparseMatrices:
        damping_diag = sp.diags(-total_damping, 0, format='csr')
        p['A'] = sp.bmat([[damping_diag, L],
                          [sp.eye(N, format='csr'), sp.csr_matrix((N, N))]]).tocsr()
        B_lil = sp.lil_matrix((2*N, 1), dtype=float)
    else:
        damping_diag = np.diag(-total_damping)
        p['A'] = np.block([[damping_diag, L],
                          [np.eye(N), np.zeros((N, N))]])
        p['B'] = np.zeros((2*N, 1))

    # =========================================================================
    # Optional: Enforce surface boundary conditions (z=0)
    # =========================================================================
    # Replicates original BC flag behavior: freezes both w and p at surface
    if enforce_surface_BC:
        # Zero out both velocity and pressure equations at surface
        # This matches the original implementation's BC flag
        if UseSparseMatrices:
            A_lil = p['A'].tolil()
            for i in range(Nx):
                k_surf_w = idx(i, 0)        # Velocity equation at surface
                k_surf_p = N + idx(i, 0)    # Pressure equation at surface
                A_lil[k_surf_w, :] = 0      # dw/dt = 0 (freeze velocity)
                A_lil[k_surf_p, :] = 0      # dp/dt = 0 (freeze pressure)
            p['A'] = A_lil.tocsr()
        else:
            for i in range(Nx):
                k_surf_w = idx(i, 0)
                k_surf_p = N + idx(i, 0)
                p['A'][k_surf_w, :] = 0
                p['A'][k_surf_p, :] = 0
    
    # =========================================================================
    # Source term (B matrix)
    # =========================================================================
    # Source applied to velocity equation at sonar location
    # Scaling by 1/(dx*dz) makes source grid-invariant
    source_idx = idx(p['sonar_ix'], p['sonar_iz'])
    
    if UseSparseMatrices:
        B_lil[source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
        p['B'] = B_lil.tocsr()
    else:
        p['B'][source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
    
    # =========================================================================
    # Initial conditions and time parameters
    # =========================================================================
    x_start = np.zeros((2*N, 1))  # Start from rest
    
    t_start = 0
    t_cross = max(Lx, Lz) / p['c']  # Acoustic crossing time
    t_stop = t_cross
    
    # CFL condition for explicit time stepping
    max_dt_FE = min(p['dx'], p['dz']) / (np.sqrt(2) * p['c']) * 0.5
    
    return p, x_start, t_start, t_stop, max_dt_FE
