import numpy as np

def eval_g_Sonar(x, p, u=None):
    """
    Output: pressure at hydrophone array
    
    Supports two formats:
    1. Original: z_pos (scalar) + x_indices (list) for horizontal array
    2. Tuple-based: x_indices + z_indices (both lists) for any configuration
    """
    N = p['Nx'] * p['Nz']
    pressure = x[:N].reshape(p['Nx'], p['Nz'])
    
    # extract pressure at each hydrophone
    hydrophone_signals = []
    
    # Try original format first (backward compatible)
    if 'z_pos' in p['hydrophones'] and 'x_indices' in p['hydrophones']:
        # Original horizontal array: same z, varying x
        z_pos = p['hydrophones']['z_pos']
        for x_idx in p['hydrophones']['x_indices']:
            if x_idx < p['Nx']:
                hydrophone_signals.append(pressure[x_idx, z_pos])
    
    else:
        # Tuple format: use x_indices and z_indices as paired lists
        # Works for vertical, horizontal, or arbitrary configurations
        
        # Handle vertical array (x_pos scalar + z_indices list)
        if 'x_pos' in p['hydrophones'] and 'z_indices' in p['hydrophones']:
            x_pos = p['hydrophones']['x_pos']
            z_indices = p['hydrophones']['z_indices']
            x_indices = [x_pos] * len(z_indices)
        
        # Handle horizontal array stored as tuple (z_pos scalar + x_indices list)  
        elif 'z_pos' in p['hydrophones'] and 'x_indices' in p['hydrophones']:
            z_pos = p['hydrophones']['z_pos']
            x_indices = p['hydrophones']['x_indices']
            z_indices = [z_pos] * len(x_indices)
        
        # Handle general case (both x_indices and z_indices as lists)
        else:
            x_indices = p['hydrophones'].get('x_indices', [])
            z_indices = p['hydrophones'].get('z_indices', [])
        
        # Extract from paired indices
        for x_idx, z_idx in zip(x_indices, z_indices):
            if x_idx < p['Nx'] and z_idx < p['Nz']:
                hydrophone_signals.append(pressure[x_idx, z_idx])
    
    return np.array(hydrophone_signals).reshape(-1, 1)  # column vector