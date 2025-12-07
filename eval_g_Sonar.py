import numpy as np

def eval_g_Sonar(x, p, u=None):
    """
    Output: pressure at hydrophone array
    Pressure is now in x[N:2N] (second half)
    """
    N = p['Nx'] * p['Nz']
    pressure = x[N:2*N].reshape(p['Nx'], p['Nz'])  # CHANGED: was x[:N]
    
    hydrophone_signals = []
    
    if 'z_pos' in p['hydrophones'] and 'x_indices' in p['hydrophones']:
        z_pos = p['hydrophones']['z_pos']
        for x_idx in p['hydrophones']['x_indices']:
            if x_idx < p['Nx']:
                hydrophone_signals.append(pressure[x_idx, z_pos])
    
    elif 'x_pos' in p['hydrophones'] and 'z_indices' in p['hydrophones']:
        x_pos = p['hydrophones']['x_pos']
        for z_idx in p['hydrophones']['z_indices']:
            if z_idx < p['Nz']:
                hydrophone_signals.append(pressure[x_pos, z_idx])
    
    return np.array(hydrophone_signals).reshape(-1, 1)