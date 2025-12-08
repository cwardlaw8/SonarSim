import numpy as np

def eval_u_Sonar_20(t):
    """Narrowband pulse - energy stays in low-frequency modes"""
    f0 = 20           # low frequency
    n_periods = 8     # more cycles = narrower bandwidth
    sigma = n_periods / (2 * f0)
    t0 = 3 * sigma
    A0 = 10
    
    if abs(t - t0) > 4 * sigma:
        return 0.0
    
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)

def eval_u_Sonar_20_const(t):
    """Continuous low-frequency ship noise with smooth ramp-on"""
    f0 = 20
    t_ramp = 0.2 #1      # smooth startup
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)

def eval_u_Sonar_3k(t):
    """
    Sonar ping with Gaussian envelope 
    """
    f0 = 3000         # Hz
    t0 = 0.0001         # s pulse center
    #sigma = 0.0001     # s pulse width
    n_periods = 3     # periods in pulse
    sigma = n_periods / (2 * f0)  
    A0 = 1            # Pa amplitude
    
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    
    # only significant within 3 sigma of center
    if abs(t - t0) > 3 * sigma:
        return 0.0
    
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)


def eval_u_Sonar_1k_const(t):
    """Continuous low-frequency ship noise with smooth ramp-on"""
    f0 = 1000
    t_ramp = 0.2      # smooth startup
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)
