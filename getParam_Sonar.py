import numpy as np
import scipy.sparse as sp

def getParam_Sonar(Nx, Nz, Lx, Lz, UseSparseMatrices=True, absorb_strength=5.0, alpha=0.001, BC=True):
    """
    State vector: x = [v_1, ..., v_N, p_1, ..., p_N]^T (velocity then pressure)
    dx/dt = A x + B u
    """
    
    p = {
        'c': 1500.0,
        'rho': 1025,
        'alpha': alpha,
        'Nx': Nx,
        'Nz': Nz,
        'Lx': Lx,
        'Lz': Lz,
        'sonar_ix': Nx//2,
        'sonar_iz': 1, 
        'absorb_strength': absorb_strength
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
    
    def idx(i, j):
        return i * Nz + j
    
    c2_dx2 = (p['c']**2) / (p['dx']**2)
    c2_dz2 = (p['c']**2) / (p['dz']**2)
    absorb_damping = p['absorb_strength'] * max(c2_dx2, c2_dz2) 
    
    # Build L matrix using COO format (fastest for construction)
    rows, cols, vals = [], [], []
    
    for i in range(Nx):
        for j in range(Nz):
            k = idx(i, j)
            
            # Skip surface nodes (j=0) if BC enabled - they stay zero
            if BC and j == 0:
                continue

            if 1 <= i < Nx - 1 and 1 <= j < Nz - 1:
                # Interior
                rows.extend([k, k, k, k, k])
                cols.extend([k, idx(i-1, j), idx(i+1, j), idx(i, j-1), idx(i, j+1)])
                vals.extend([-2*(c2_dx2 + c2_dz2), c2_dx2, c2_dx2, c2_dz2, c2_dz2])
            
            elif i == 0 and 1 <= j < Nz - 1:
                # Left boundary (absorbing)
                rows.extend([k, k, k, k])
                cols.extend([k, idx(i+1, j), idx(i, j-1), idx(i, j+1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2 - absorb_damping, 2*c2_dx2, c2_dz2, c2_dz2])
            
            elif i == Nx - 1 and 1 <= j < Nz - 1:
                # Right boundary (absorbing)
                rows.extend([k, k, k, k])
                cols.extend([k, idx(i-1, j), idx(i, j-1), idx(i, j+1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2 - absorb_damping, 2*c2_dx2, c2_dz2, c2_dz2])
            
            elif j == Nz - 1 and 1 <= i < Nx - 1:
                # Bottom boundary (seafloor - rigid)
                rows.extend([k, k, k, k])
                cols.extend([k, idx(i-1, j), idx(i+1, j), idx(i, j-1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2, c2_dx2, c2_dx2, 2*c2_dz2])

            elif i == 0 and j == Nz-1:
                # Bottom-left corner
                rows.extend([k, k, k])
                cols.extend([k, idx(i+1, j), idx(i, j-1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2 - absorb_damping, 2*c2_dx2, 2*c2_dz2])
            
            elif i == Nx-1 and j == Nz-1:
                # Bottom-right corner
                rows.extend([k, k, k])
                cols.extend([k, idx(i-1, j), idx(i, j-1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2 - absorb_damping, 2*c2_dx2, 2*c2_dz2])
    
    # Build sparse L from COO data
    if UseSparseMatrices:
        L = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    else:
        L = np.zeros((N, N))
        for r, c, v in zip(rows, cols, vals):
            L[r, c] = v
    
    # Build A matrix: A = [[-αI, L], [I, 0]]
    # dv/dt = -αv + Lp
    # dp/dt = v
    if UseSparseMatrices:
        # Build diagonal for velocity damping, zeroing surface nodes if BC
        diag_v = -p['alpha'] * np.ones(N)
        if BC:
            for i in range(Nx):
                diag_v[i * Nz] = 0  # surface nodes
        
        # Build identity for dp/dt = v, zeroing surface nodes if BC
        diag_I = np.ones(N)
        if BC:
            for i in range(Nx):
                diag_I[i * Nz] = 0  # surface nodes
        
        p['A'] = sp.bmat([
            [sp.diags(diag_v), L],
            [sp.diags(diag_I), sp.csr_matrix((N, N))]
        ]).tocsr()
        
        B_lil = sp.lil_matrix((2*N, 1), dtype=float)
    else:
        p['A'] = np.block([
            [-p['alpha']*np.eye(N), L],
            [np.eye(N), np.zeros((N, N))]
        ])
        if BC:
            for i in range(Nx):
                k = i * Nz
                p['A'][k, k] = 0      # zero damping at surface
                p['A'][N + k, k] = 0  # dp/dt = 0 at surface
        p['B'] = np.zeros((2*N, 1))
    
    # Source drives velocity (first N entries)
    source_idx = idx(p['sonar_ix'], p['sonar_iz'])
    if UseSparseMatrices:
        B_lil[source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
        p['B'] = B_lil.tocsr()
    else:
        p['B'][source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
    
    x_start = np.zeros((2*N, 1))
    
    t_start = 0
    t_cross = max(Lx, Lz) / p['c']
    t_stop = t_cross
    
    max_dt_FE = min(p['dx'], p['dz']) / (np.sqrt(2) * p['c']) * 0.5
    
    return p, x_start, t_start, t_stop, max_dt_FE



def getParam_Sonar_center(Nx, Nz, Lx, Lz, UseSparseMatrices=True, absorb_strength=5.0, alpha=0.001, BC=True):
    """
    State vector: x = [v_1, ..., v_N, p_1, ..., p_N]^T (velocity then pressure)
    dx/dt = A x + B u
    """
    
    p = {
        'c': 1500.0,
        'rho': 1025,
        'alpha': alpha,
        'Nx': Nx,
        'Nz': Nz,
        'Lx': Lx,
        'Lz': Lz,
        'sonar_ix': Nx//2,
        'sonar_iz': Nz//2, 
        'absorb_strength': absorb_strength
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
    
    def idx(i, j):
        return i * Nz + j
    
    c2_dx2 = (p['c']**2) / (p['dx']**2)
    c2_dz2 = (p['c']**2) / (p['dz']**2)
    absorb_damping = p['absorb_strength'] * max(c2_dx2, c2_dz2) 
    
    # Build L matrix using COO format (fastest for construction)
    rows, cols, vals = [], [], []
    
    for i in range(Nx):
        for j in range(Nz):
            k = idx(i, j)
            
            # Skip surface nodes (j=0) if BC enabled - they stay zero
            if BC and j == 0:
                continue

            if 1 <= i < Nx - 1 and 1 <= j < Nz - 1:
                # Interior
                rows.extend([k, k, k, k, k])
                cols.extend([k, idx(i-1, j), idx(i+1, j), idx(i, j-1), idx(i, j+1)])
                vals.extend([-2*(c2_dx2 + c2_dz2), c2_dx2, c2_dx2, c2_dz2, c2_dz2])
            
            elif i == 0 and 1 <= j < Nz - 1:
                # Left boundary (absorbing)
                rows.extend([k, k, k, k])
                cols.extend([k, idx(i+1, j), idx(i, j-1), idx(i, j+1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2 - absorb_damping, 2*c2_dx2, c2_dz2, c2_dz2])
            
            elif i == Nx - 1 and 1 <= j < Nz - 1:
                # Right boundary (absorbing)
                rows.extend([k, k, k, k])
                cols.extend([k, idx(i-1, j), idx(i, j-1), idx(i, j+1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2 - absorb_damping, 2*c2_dx2, c2_dz2, c2_dz2])
            
            elif j == Nz - 1 and 1 <= i < Nx - 1:
                # Bottom boundary (seafloor - rigid)
                rows.extend([k, k, k, k])
                cols.extend([k, idx(i-1, j), idx(i+1, j), idx(i, j-1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2, c2_dx2, c2_dx2, 2*c2_dz2])

            elif i == 0 and j == Nz-1:
                # Bottom-left corner
                rows.extend([k, k, k])
                cols.extend([k, idx(i+1, j), idx(i, j-1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2 - absorb_damping, 2*c2_dx2, 2*c2_dz2])
            
            elif i == Nx-1 and j == Nz-1:
                # Bottom-right corner
                rows.extend([k, k, k])
                cols.extend([k, idx(i-1, j), idx(i, j-1)])
                vals.extend([-2*c2_dx2 - 2*c2_dz2 - absorb_damping, 2*c2_dx2, 2*c2_dz2])
    
    # Build sparse L from COO data
    if UseSparseMatrices:
        L = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    else:
        L = np.zeros((N, N))
        for r, c, v in zip(rows, cols, vals):
            L[r, c] = v
    
    # Build A matrix: A = [[-αI, L], [I, 0]]
    # dv/dt = -αv + Lp
    # dp/dt = v
    if UseSparseMatrices:
        # Build diagonal for velocity damping, zeroing surface nodes if BC
        diag_v = -p['alpha'] * np.ones(N)
        if BC:
            for i in range(Nx):
                diag_v[i * Nz] = 0  # surface nodes
        
        # Build identity for dp/dt = v, zeroing surface nodes if BC
        diag_I = np.ones(N)
        if BC:
            for i in range(Nx):
                diag_I[i * Nz] = 0  # surface nodes
        
        p['A'] = sp.bmat([
            [sp.diags(diag_v), L],
            [sp.diags(diag_I), sp.csr_matrix((N, N))]
        ]).tocsr()
        
        B_lil = sp.lil_matrix((2*N, 1), dtype=float)
    else:
        p['A'] = np.block([
            [-p['alpha']*np.eye(N), L],
            [np.eye(N), np.zeros((N, N))]
        ])
        if BC:
            for i in range(Nx):
                k = i * Nz
                p['A'][k, k] = 0      # zero damping at surface
                p['A'][N + k, k] = 0  # dp/dt = 0 at surface
        p['B'] = np.zeros((2*N, 1))
    
    # Source drives velocity (first N entries)
    source_idx = idx(p['sonar_ix'], p['sonar_iz'])
    if UseSparseMatrices:
        B_lil[source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
        p['B'] = B_lil.tocsr()
    else:
        p['B'][source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
    
    x_start = np.zeros((2*N, 1))
    
    t_start = 0
    t_cross = max(Lx, Lz) / p['c']
    t_stop = t_cross
    
    max_dt_FE = min(p['dx'], p['dz']) / (np.sqrt(2) * p['c']) * 0.5
    
    return p, x_start, t_start, t_stop, max_dt_FE





'''
def getParam_Sonar(Nx, Nz, Lx, Lz, UseSparseMatrices=True, absorb_strength=5.0, alpha=0.001, BC=True):
    """
    State vector: x = [v_1, ..., v_N, p_1, ..., p_N]^T (velocity then pressure)
    dx/dt = A x + B u
    """
    
    p = {
        'c': 1500.0,
        'rho': 1025,
        'alpha': alpha,
        'Nx': Nx,
        'Nz': Nz,
        'Lx': Lx,
        'Lz': Lz,
        'sonar_ix': Nx//2,
        'sonar_iz': Nz//2,
        'absorb_strength': absorb_strength
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

            if 1 <= i < Nx - 1 and 1 <= j < Nz - 1:
                L[k, k] = -2 * (c2_dx2 + c2_dz2)
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2
            
            elif i == 0 and 1 <= j < Nz - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i+1, j)] = 2*c2_dx2
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2
            
            elif i == Nx - 1 and 1 <= j < Nz - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i-1, j)] = 2*c2_dx2
                L[k, idx(i, j-1)] = c2_dz2
                L[k, idx(i, j+1)] = c2_dz2
            
            elif j == 0 and 1 <= i < Nx - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j+1)] = 2*c2_dz2
            
            elif j == Nz - 1 and 1 <= i < Nx - 1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2
                L[k, idx(i-1, j)] = c2_dx2
                L[k, idx(i+1, j)] = c2_dx2
                L[k, idx(i, j-1)] = 2*c2_dz2

            elif i == 0 and j == 0:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i+1, j)] = 2*c2_dx2
                L[k, idx(i, j+1)] = 2*c2_dz2
            
            elif i == Nx-1 and j == 0:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i-1, j)] = 2*c2_dx2
                L[k, idx(i, j+1)] = 2*c2_dz2
            
            elif i == 0 and j == Nz-1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i+1, j)] = 2*c2_dx2
                L[k, idx(i, j-1)] = 2*c2_dz2
            
            elif i == Nx-1 and j == Nz-1:
                L[k, k] = -2*c2_dx2 - 2*c2_dz2 - absorb_damping
                L[k, idx(i-1, j)] = 2*c2_dx2
                L[k, idx(i, j-1)] = 2*c2_dz2
    
    # A = [[-αI, L], [I, 0]]
    # dv/dt = -αv + Lp
    # dp/dt = v
    if UseSparseMatrices:
        L = L.tocsr()
        p['A'] = sp.bmat([[-p['alpha']*sp.eye(N), L],
                          [sp.eye(N), sp.csr_matrix((N, N))]]).tocsr()
        B_lil = sp.lil_matrix((2*N, 1), dtype=float)
    else:
        p['A'] = np.block([[-p['alpha']*np.eye(N), L],
                          [np.eye(N), np.zeros((N, N))]])
        p['B'] = np.zeros((2*N, 1))

    if BC:
        # Enforce p=0 at surface (j=0)
        if UseSparseMatrices:
            A_lil = p['A'].tolil()
            for i in range(Nx):
                k = i * Nz  # surface node (j=0)
                A_lil[k, :] = 0          # dv/dt = 0
                A_lil[N + k, :] = 0      # dp/dt = 0
            p['A'] = A_lil.tocsr()
        else:
            for i in range(Nx):
                k = i * Nz
                p['A'][k, :] = 0
                p['A'][N + k, :] = 0
    
    # Source drives velocity (first N entries)
    source_idx = idx(p['sonar_ix'], p['sonar_iz'])
    if UseSparseMatrices:
        B_lil[source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
        p['B'] = B_lil.tocsr()
    else:
        p['B'][source_idx, 0] = 1.0 / (p['dx'] * p['dz'])
    
    x_start = np.zeros((2*N, 1))
    
    t_start = 0
    t_cross = max(Lx, Lz) / p['c']
    t_stop = t_cross
    
    max_dt_FE = min(p['dx'], p['dz']) / (np.sqrt(2) * p['c']) * 0.5
    
    return p, x_start, t_start, t_stop, max_dt_FE
'''