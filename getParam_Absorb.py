"""
Absorbing Boundary Conditions for 2D Acoustic Wave Equation

Two implementations for comparison:
1. Sponge Layer: Simple damping that increases toward boundaries
2. PML (Perfectly Matched Layer): Complex coordinate stretching for reflection-free absorption

Both return systems in the form: dx/dt = A @ x + B @ u
"""

import numpy as np
import scipy.sparse as sp


# =============================================================================
# SPONGE LAYER IMPLEMENTATION
# =============================================================================

def create_sponge_profile(Nx, Nz, sponge_width, max_damping, profile='cubic'):
    """
    Create smooth sponge damping profile for absorbing boundaries.
    """
    exponents = {'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4}
    m = exponents.get(profile, 3)
    
    damping = np.zeros((Nx, Nz))
    
    for i in range(Nx):
        if i < sponge_width:
            d = (sponge_width - i) / sponge_width
            damping[i, :] = max_damping * (d ** m)
        elif i >= Nx - sponge_width:
            d = (i - (Nx - sponge_width) + 1) / sponge_width
            damping[i, :] = max_damping * (d ** m)
    
    return damping


def getParam_Sponge(Nx, Nz, Lx, Lz, UseSparseMatrices=True, c=1500.0, rho=1025.0, 
                    alpha=0.1, sponge_width_frac=0.1, sponge_max_damping=None,
                    sponge_profile='cubic'):
    """
    2D acoustic wave equation with SPONGE LAYER absorbing boundaries.
    
    State vector: x = [p; v] (2N variables)
    """
    # Grid setup
    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)
    N = Nx * Nz
    
    # Sponge configuration
    sponge_width = max(10, int(sponge_width_frac * Nx))
    if sponge_max_damping is None:
        sponge_max_damping = 3.0 * c / min(dx, dz)
    
    sponge_damping = create_sponge_profile(Nx, Nz, sponge_width, 
                                            sponge_max_damping, sponge_profile)
    
    # Build Laplacian with boundary conditions
    L = sp.lil_matrix((N, N), dtype=float)
    
    def idx(i, j):
        return i * Nz + j
    
    c2_dx2 = (c**2) / (dx**2)
    c2_dz2 = (c**2) / (dz**2)
    
    for i in range(Nx):
        for j in range(Nz):
            k = idx(i, j)
            
            at_left = (i == 0)
            at_right = (i == Nx - 1)
            at_top = (j == 0)
            at_bottom = (j == Nz - 1)
            
            L[k, k] = -2 * (c2_dx2 + c2_dz2)
            
            # X-direction stencil
            if at_left:
                L[k, idx(i+1, j)] = 2 * c2_dx2
            elif at_right:
                L[k, idx(i-1, j)] = 2 * c2_dx2
            else:
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
            
            # Z-direction stencil
            if at_top:
                L[k, idx(i, j+1)] = 2 * c2_dz2  # Pressure release
            elif at_bottom:
                L[k, idx(i, j-1)] = 2 * c2_dz2  # Rigid
            else:
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2
    
    # Total damping = global + sponge
    total_damping = alpha + sponge_damping.flatten(order='C')
    
    # Source location
    sonar_ix, sonar_iz = Nx // 4, Nz // 2
    source_idx = idx(sonar_ix, sonar_iz)
    
    # Build state-space system
    if UseSparseMatrices:
        D = sp.diags(-total_damping, 0, format='csr')
        A = sp.bmat([
            [sp.csr_matrix((N, N)), sp.eye(N, format='csr')],
            [L.tocsr(), D]
        ]).tocsr()
        B = sp.lil_matrix((2*N, 1), dtype=float)
        B[N + source_idx, 0] = 1.0 / (dx * dz)
        B = B.tocsr()
    else:
        L = L.toarray()
        D = np.diag(-total_damping)
        A = np.block([
            [np.zeros((N, N)), np.eye(N)],
            [L, D]
        ])
        B = np.zeros((2*N, 1))
        B[N + source_idx, 0] = 1.0 / (dx * dz)
    
    # Hydrophone array setup
    n_phones = 5
    spacing = max(1, (Nx - Nx//2) // (n_phones - 1))
    hydrophones = {
        'z_pos': Nz // 2,
        'x_indices': [Nx//4 + i*spacing for i in range(n_phones)],
        'n_phones': n_phones
    }
    
    # Package parameters
    p = {
        'A': A, 'B': B,
        'c': c, 'rho': rho, 'alpha': alpha,
        'Nx': Nx, 'Nz': Nz, 'N': N,
        'Lx': Lx, 'Lz': Lz,
        'dx': dx, 'dz': dz,
        'sonar_ix': sonar_ix, 'sonar_iz': sonar_iz,
        'hydrophones': hydrophones,
        'boundary_type': 'sponge',
        'sponge_width': sponge_width,
        'sponge_max_damping': sponge_max_damping,
        'sponge_profile': sponge_profile,
        'sponge_damping': sponge_damping,
    }
    
    # Initial conditions
    x0 = np.zeros((2*N, 1))
    rng = np.random.default_rng(0)
    x0[:N] = rng.standard_normal((N, 1)) * 0
    
    t_start = 0.0
    t_stop = max(Lx, Lz) / c
    max_dt = 0.5 * min(dx, dz) / (np.sqrt(2) * c)
    
    return p, x0, t_start, t_stop, max_dt


# =============================================================================
# PML IMPLEMENTATION
# =============================================================================

def create_pml_profile(Nx, Nz, pml_width, sigma_max, profile='cubic'):
    """Create PML damping profiles for x-direction boundaries."""
    exponents = {'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4}
    m = exponents.get(profile, 3)
    
    sigma_x = np.zeros((Nx, Nz))
    
    for i in range(Nx):
        if i < pml_width:
            d = (pml_width - i) / pml_width
            sigma_x[i, :] = sigma_max * (d ** m)
        elif i >= Nx - pml_width:
            d = (i - (Nx - pml_width) + 1) / pml_width
            sigma_x[i, :] = sigma_max * (d ** m)
    
    return sigma_x


def getParam_PML(Nx, Nz, Lx, Lz, UseSparseMatrices=True, c=1500.0, rho=1025.0, 
                 alpha=0.1, pml_width_frac=0.1, pml_sigma_max=None, pml_profile='cubic'):
    """
    2D acoustic wave equation with PML (Perfectly Matched Layer) boundaries.
    
    State vector: x = [p; vx; vz; ψ] (4N variables)
    """
    # Grid setup
    dx = Lx / (Nx - 1)
    dz = Lz / (Nz - 1)
    N = Nx * Nz
    
    # PML configuration
    pml_width = max(10, int(pml_width_frac * Nx))
    
    if pml_sigma_max is None:
        pml_thickness = pml_width * dx
        m = {'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4}[pml_profile]
        pml_sigma_max = (m + 1) * c * np.log(1e6) / (2 * pml_thickness)
    
    sigma_x = create_pml_profile(Nx, Nz, pml_width, pml_sigma_max, pml_profile)
    sigma_x_flat = sigma_x.flatten(order='C')
    
    def idx(i, j):
        return i * Nz + j
    
    # Build spatial derivative operators
    Dx = sp.lil_matrix((N, N), dtype=float)
    Dz = sp.lil_matrix((N, N), dtype=float)
    
    for i in range(Nx):
        for j in range(Nz):
            k = idx(i, j)
            
            # X-derivative
            if i == 0:
                Dx[k, idx(i, j)] = -1.0 / dx
                Dx[k, idx(i+1, j)] = 1.0 / dx
            elif i == Nx - 1:
                Dx[k, idx(i-1, j)] = -1.0 / dx
                Dx[k, idx(i, j)] = 1.0 / dx
            else:
                Dx[k, idx(i-1, j)] = -0.5 / dx
                Dx[k, idx(i+1, j)] = 0.5 / dx
            
            # Z-derivative
            if j == 0:
                Dz[k, idx(i, j)] = -1.0 / dz
                Dz[k, idx(i, j+1)] = 1.0 / dz
            elif j == Nz - 1:
                Dz[k, idx(i, j-1)] = -1.0 / dz
                Dz[k, idx(i, j)] = 1.0 / dz
            else:
                Dz[k, idx(i, j-1)] = -0.5 / dz
                Dz[k, idx(i, j+1)] = 0.5 / dz
    
    Dx = Dx.tocsr()
    Dz = Dz.tocsr()
    
    # Build PML system matrix
    Z = sp.csr_matrix((N, N))
    
    Sigma_x = sp.diags(sigma_x_flat, 0, format='csr')
    Alpha = sp.diags(np.full(N, alpha), 0, format='csr')
    Alpha_Sigma = Alpha + Sigma_x
    
    rho_c2 = rho * c**2
    inv_rho = 1.0 / rho
    
    # State: [p; vx; vz; ψ]
    A = sp.bmat([
        [-Alpha_Sigma,      -rho_c2 * Dx, -rho_c2 * Dz, Sigma_x],
        [-inv_rho * Dx,     -Alpha_Sigma, Z,            Z],
        [-inv_rho * Dz,     Z,            -Alpha,       Z],
        [Z,                 Z,            -rho_c2 * Dz, -Alpha_Sigma]
    ]).tocsr()
    
    # Source injection
    sonar_ix, sonar_iz = Nx // 4, Nz // 2
    source_idx = idx(sonar_ix, sonar_iz)
    B = sp.lil_matrix((4*N, 1), dtype=float)
    B[source_idx, 0] = 1.0 / (dx * dz)
    B = B.tocsr()
    
    if not UseSparseMatrices:
        A = A.toarray()
        B = B.toarray()
    
    # Hydrophone array setup
    n_phones = 5
    spacing = max(1, (Nx - Nx//2) // (n_phones - 1))
    hydrophones = {
        'z_pos': Nz // 2,
        'x_indices': [Nx//4 + i*spacing for i in range(n_phones)],
        'n_phones': n_phones
    }
    
    # Package parameters
    p = {
        'A': A, 'B': B,
        'c': c, 'rho': rho, 'alpha': alpha,
        'Nx': Nx, 'Nz': Nz, 'N': N,
        'Lx': Lx, 'Lz': Lz,
        'dx': dx, 'dz': dz,
        'sonar_ix': sonar_ix, 'sonar_iz': sonar_iz,
        'hydrophones': hydrophones,
        'boundary_type': 'pml',
        'pml_width': pml_width,
        'pml_sigma_max': pml_sigma_max,
        'pml_profile': pml_profile,
        'sigma_x': sigma_x,
        'state_size': 4*N,
        'state_layout': {'p': (0, N), 'vx': (N, 2*N), 'vz': (2*N, 3*N), 'psi': (3*N, 4*N)},
    }
    
    # Initial conditions
    x0 = np.zeros((4*N, 1))
    rng = np.random.default_rng(0)
    x0[:N] = rng.standard_normal((N, 1)) * 0
    
    t_start = 0.0
    t_stop = max(Lx, Lz) / c
    max_dt = 0.5 * min(dx, dz) / (np.sqrt(2) * c)
    
    return p, x0, t_start, t_stop, max_dt


# =============================================================================
# UTILITIES
# =============================================================================

def extract_pressure(x, p):
    """Extract pressure field from state vector for either method."""
    N = p['N']
    return x[:N].reshape((p['Nx'], p['Nz']), order='C')