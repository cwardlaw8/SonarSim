import numpy as np
import scipy.sparse as sp

def getParam_Sonar(Nx, Nz, Lx, Lz, UseSparseMatrices=True, absorb_strength=5.0, alpha=0.001, BC=False):
    """
    Defines the parameters for 2D acoustic wave equation for sonar propagation.
    Returns matrices for the linear system representation dx/dt = p.A x + p.B u
    where state x = [p_1, ..., p_N, v_1, ..., v_N]^T (pressure and velocity)

    INPUT:
    Nx         number of grid points in x direction
    Nz         number of grid points in z direction
    Lx         total length in x direction 
    Lz         total length in z direction 

    OUPUTS:
    p.A         system matrix (2Nx2N)
    p.B         input matrix (2Nx1)
    p.c         speed of sound
    p.rho       density of the medium
    p.alpha     absorption coefficient
    p.dx        spatial step in x direction
    p.dz        spatial step in z direction
    p.sonar_ix  sonar source grid index in x direction
    p.sonar_iz  sonar source grid index in z direction

    x_start     initial state vector
    t_start     initial time
    t_stop      simulation end time
    max_dt_FE   maximum stable timestep for Forward Euler

    EXAMPLE:
    [p,x_start,t_start,t_stop,max_dt_FE] = getParam_Sonar(Nx, Nz, Lx, Lz);
    """
    
    p = {
        'c': 1500.0,           # (m/s) speed of sound
        'rho': 1025,           # kg/m^3 density
        'alpha': alpha,        # (1/s) very weak global absorption
        'Nx': Nx,              # grid points in x
        'Nz': Nz,              # grid points in z
        'Lx': Lx,              # domain size x (m)
        'Lz': Lz,              # domain size z (m)
        'sonar_ix': Nx//2,     # source position x
        'sonar_iz': 1,         # source position z
        'absorb_strength': absorb_strength # strength of absorbing boundary
    }
    
    n_phones = 5
    p['hydrophones'] = {
        'z_pos': Nz // 2,
        'x_indices': np.linspace(Nx//8, Nx - Nx//8, n_phones, dtype=int).tolist(),
        'n_phones': n_phones
    }

    p['dx'] = Lx / (Nx - 1)
    p['dz'] = Lz / (Nz - 1)
    
    N = Nx * Nz
    
    if UseSparseMatrices:
        L = sp.lil_matrix((N, N), dtype=float)
    else:
        L = np.zeros((N, N))
    
    def idx(i, j):
        return i * Nz + j
    
    c2_dx2 = (p['c']**2) / (p['dx']**2)
    c2_dz2 = (p['c']**2) / (p['dz']**2)
     
    absorb_damping = p['absorb_strength'] * max(c2_dx2, c2_dz2) 
    
    for i in range(Nx):
        for j in range(Nz):
            k = idx(i, j)

            # interior
            if 1 <= i < Nx - 1 and 1 <= j < Nz - 1:
                L[k, k] = -2 * (c2_dx2 + c2_dz2)
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2
            
            # left boundary (x=0): absorbing
            # Add damping directly to the diagonal
            elif i == 0 and 1 <= j < Nz - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i+1, j)] = 2*c2_dx2  # one-sided difference
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2
            
            # right boundary (x=Lx): absorbing
            elif i == Nx - 1 and 1 <= j < Nz - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i-1, j)] = 2*c2_dx2  # one-sided difference
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2
            
            # top boundary (z=0): pressure release (sea Surface)
            # ghost point method: p_ghost = -p_interior for p=0 at boundary
            elif j == 0 and 1 <= i < Nx - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j+1)] = 2*c2_dz2  # double weight for pressure-release
            
            # bottom boundary (z=Lz): rigid (seafloor)
            # ghost point method: p_ghost = p_interior for dp/dn=0
            elif j == Nz - 1 and 1 <= i < Nx - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j-1)] = 2*c2_dz2  # double weight for rigid
            

            # corners
            # top-left corner (pressure-release top + absorbing left)
            elif i == 0 and j == 0:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i+1, j)] = 2*c2_dx2  # absorbing
                L[k, idx(i, j+1)] = 2*c2_dz2  # pressure-release
            
            # top-right corner (pressure-release top + absorbing right)
            elif i == Nx-1 and j == 0:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i-1, j)] = 2*c2_dx2  # absorbing
                L[k, idx(i, j+1)] = 2*c2_dz2  # pressure-release
            
            # bottom-left corner (rigid bottom + absorbing left)
            elif i == 0 and j == Nz-1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i+1, j)] = 2*c2_dx2  # absorbing
                L[k, idx(i, j-1)] = 2*c2_dz2  # rigid
            
            # bottom-right corner (rigid bottom + absorbing right)
            elif i == Nx-1 and j == Nz-1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i-1, j)] = 2*c2_dx2  # absorbing
                L[k, idx(i, j-1)] = 2*c2_dz2  # rigid
    
    if UseSparseMatrices:
        p['A'] = sp.bmat([[sp.csr_matrix((N, N)), sp.eye(N)],
                          [L, -p['alpha']*sp.eye(N)]]).tocsr()
        B_lil = sp.lil_matrix((2*N, 1), dtype=float)
    else:
        p['A'] = np.block([[np.zeros((N, N)), np.eye(N)],
                          [L, -p['alpha']*np.eye(N)]])
        p['B'] = np.zeros((2*N, 1))

    if BC:
        # Enforce p=0 at surface (j=0) - Dirichlet boundary condition
        if UseSparseMatrices:
            A_lil = p['A'].tolil() # type: ignore
            for i in range(Nx):
                k = i * Nz  # surface node (j=0)
                A_lil[k, :] = 0      # dp/dt = 0
                A_lil[N + k, :] = 0  # dv/dt = 0
            p['A'] = A_lil.tocsr()
        else:
            for i in range(Nx):
                k = i * Nz
                p['A'][k, :] = 0
                p['A'][N + k, :] = 0
    
    # source location
    # Scale source by cell area so the effective source is grid-invariant.
    # With this choice, Bu has units of [Pa/s^2] provided u(t) has units [Pa·m^2/s^2].
    source_idx = idx(p['sonar_ix'], p['sonar_iz'])
    if UseSparseMatrices:
        B_lil[N + source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
        p['B'] = B_lil.tocsr()
    else:
        p['B'][N + source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
    
    # initial conditions
    x_start = np.zeros((2*N, 1))
    # small deterministic noise in pressure for reproducible visuals
    #rng = np.random.default_rng(0)
    #x_start[:N] = rng.standard_normal((N, 1)) * 1e-10
    #x_start[:N] = rng.standard_normal((N, 1)) * 0
    
    t_start = 0
    t_cross = max(Lx, Lz) / p['c']
    t_stop = t_cross
    
    # CFL condition for stability
    max_dt_FE = min(p['dx'], p['dz']) / (np.sqrt(2) * p['c']) * 0.5
    
    return p, x_start, t_start, t_stop, max_dt_FE
