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

# sources for sweeping


def eval_u_20_const_half_amp(t):
    """Continuous 20 Hz at half amplitude"""
    return 0.5 * eval_u_Sonar_20_const(t)


def eval_u_20_const_double_amp(t):
    """Continuous 20 Hz at double amplitude"""
    return 2.0 * eval_u_Sonar_20_const(t)


def eval_u_20_pulse_short(t):
    """Short Gaussian pulse at 20 Hz (2 cycles)"""
    f0 = 20
    n_periods = 2
    sigma = n_periods / (2 * f0)
    t0 = 3 * sigma
    A0 = 100
    
    if abs(t - t0) > 4 * sigma:
        return 0.0
    
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)

def eval_u_20_pulse_long(t):
    """Long Gaussian pulse at 20 Hz (8 cycles)"""
    f0 = 20
    n_periods = 8
    sigma = n_periods / (2 * f0)
    t0 = 3 * sigma
    A0 = 100
    
    if abs(t - t0) > 4 * sigma:
        return 0.0
    
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)


def eval_u_20_pulse_hann(t):
    """Hann-windowed pulse at 20 Hz"""
    f0 = 20
    t_duration = 0.25  # 5 cycles
    A0 = 100
    
    if t < 0 or t > t_duration:
        return 0.0
    
    envelope = 0.5 * (1 - np.cos(2 * np.pi * t / t_duration))
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)


def eval_u_20_pulse_rect(t):
    """Rectangular pulse at 20 Hz (abrupt on/off)"""
    f0 = 20
    t_start = 0.05
    t_end = 0.3
    A0 = 100
    
    if t < t_start or t > t_end:
        return 0.0
    
    return A0 * np.sin(2 * np.pi * f0 * t)

def eval_u_20_AM_slow(t):
    """Amplitude modulated 20 Hz (0.5 Hz modulation)"""
    f0 = 20
    f_mod = 0.5
    t_ramp = 0.2
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    modulation = 1 + 0.5 * np.sin(2 * np.pi * f_mod * t)
    return A0 * envelope * modulation * np.sin(2 * np.pi * f0 * t)


def eval_u_20_AM_fast(t):
    """Amplitude modulated 20 Hz (2 Hz modulation)"""
    f0 = 20
    f_mod = 2
    t_ramp = 0.2
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    modulation = 1 + 0.5 * np.sin(2 * np.pi * f_mod * t)
    return A0 * envelope * modulation * np.sin(2 * np.pi * f0 * t)


def eval_u_20_FM(t):
    """Frequency modulated around 20 Hz (±2 Hz deviation)"""
    f0 = 20
    f_mod = 1
    f_dev = 2  # ±2 Hz deviation
    t_ramp = 0.2
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    # Instantaneous phase
    phase = 2 * np.pi * f0 * t + (f_dev / f_mod) * np.sin(2 * np.pi * f_mod * t)
    return A0 * envelope * np.sin(phase)

def eval_u_chirp_up(t):
    """Upward chirp: 15 Hz to 25 Hz"""
    f_start, f_end = 15, 25
    t_ramp = 0.1
    t_chirp = 1.0  # chirp duration
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    elif t > t_chirp:
        envelope = 0.0
    else:
        envelope = 1.0
    
    # Linear chirp
    f_inst = f_start + (f_end - f_start) * t / t_chirp
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / t_chirp)
    return A0 * envelope * np.sin(phase)


def eval_u_chirp_down(t):
    """Downward chirp: 25 Hz to 15 Hz"""
    f_start, f_end = 25, 15
    t_ramp = 0.1
    t_chirp = 1.0
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    elif t > t_chirp:
        envelope = 0.0
    else:
        envelope = 1.0
    
    f_inst = f_start + (f_end - f_start) * t / t_chirp
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / t_chirp)
    return A0 * envelope * np.sin(phase)


def eval_u_chirp_narrow(t):
    """Narrow chirp: 19 Hz to 21 Hz"""
    f_start, f_end = 19, 21
    t_ramp = 0.1
    t_chirp = 1.0
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    elif t > t_chirp:
        envelope = 0.0
    else:
        envelope = 1.0
    
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / t_chirp)
    return A0 * envelope * np.sin(phase)


def eval_u_20_burst(t):
    """Repeated bursts at 20 Hz (on/off)"""
    f0 = 20
    A0 = 100
    burst_on = 0.15
    burst_off = 0.1
    period = burst_on + burst_off
    
    t_in_period = t % period
    if t_in_period < burst_on:
        return A0 * np.sin(2 * np.pi * f0 * t)
    return 0.0


def eval_u_20_burst_fade(t):
    """Repeated bursts with fade in/out"""
    f0 = 20
    A0 = 100
    burst_duration = 0.2
    gap = 0.1
    period = burst_duration + gap
    
    t_in_period = t % period
    if t_in_period < burst_duration:
        # Smooth envelope within burst
        envelope = np.sin(np.pi * t_in_period / burst_duration)
        return A0 * envelope * np.sin(2 * np.pi * f0 * t)
    return 0.0


def eval_u_20_double_pulse(t):
    """Two separated pulses"""
    f0 = 20
    A0 = 100
    pulse_duration = 0.15
    
    # First pulse: t = 0.05 to 0.2
    # Second pulse: t = 0.4 to 0.55
    if 0.05 < t < 0.05 + pulse_duration:
        envelope = np.sin(np.pi * (t - 0.05) / pulse_duration)
        return A0 * envelope * np.sin(2 * np.pi * f0 * t)
    elif 0.4 < t < 0.4 + pulse_duration:
        envelope = np.sin(np.pi * (t - 0.4) / pulse_duration)
        return A0 * envelope * np.sin(2 * np.pi * f0 * t)
    return 0.0

def eval_u_20_plus_harmonic(t):
    """20 Hz + weak 40 Hz harmonic"""
    f1, f2 = 20, 40
    A1, A2 = 100, 20  # Harmonic at 20% amplitude
    t_ramp = 0.2
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    return envelope * (A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t))


def eval_u_20_plus_subharmonic(t):
    """20 Hz + weak 10 Hz subharmonic"""
    f1, f2 = 20, 10
    A1, A2 = 100, 30
    t_ramp = 0.2
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    return envelope * (A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t))

def eval_u_100_const(t):
    """100 Hz - higher frequency, smaller wavelength"""
    f0 = 100
    t_ramp = 0.1
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)


def eval_u_5_const(t):
    """5 Hz - lower frequency, larger wavelength"""
    f0 = 5
    t_ramp = 0.5
    A0 = 100
    
    if t < t_ramp:
        envelope = 0.5 * (1 - np.cos(np.pi * t / t_ramp))
    else:
        envelope = 1.0
    
    return A0 * envelope * np.sin(2 * np.pi * f0 * t)

