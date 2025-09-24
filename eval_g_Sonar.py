import numpy as np

def eval_g_Sonar(x, p, u=None):
    """
    Output: pressure at hydrophone array
    """
    N = p['Nx'] * p['Nz']
    pressure = x[:N].reshape(p['Nx'], p['Nz'])
    
    # extract pressure at each hydrophone
    hydrophone_signals = []
    z_pos = p['hydrophones']['z_pos']
    
    for x_idx in p['hydrophones']['x_indices']:
        if x_idx < p['Nx']:
            hydrophone_signals.append(pressure[x_idx, z_pos])
    
    return np.array(hydrophone_signals).reshape(-1, 1) # column vector