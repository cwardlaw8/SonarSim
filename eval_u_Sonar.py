import numpy as np

def eval_u_Sonar_coarse(t):
    """
    Sonar ping with Gaussian envelope 
    """
    f0 = 20            # Hz
    t0 = 0.002          # s pulse center
    n_periods = 3     # periods in pulse
    sigma = n_periods / (2 * f0)  
    A0 = 1         # Pa amplitude

    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    
    # only significant within 4 sigma of center
    if abs(t - t0) > 4 * sigma:
        return 0.0
    
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)


def eval_u_Sonar(t):
    """
    Sonar ping with Gaussian envelope 
    """
    f0 = 3000         # Hz
    t0 = 0.0001         # s pulse center
    sigma = 0.0001     # s pulse width
    A0 = 1            # Pa amplitude
    
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    
    # only significant within 3 sigma of center
    if abs(t - t0) > 3 * sigma:
        return 0.0
    
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)
