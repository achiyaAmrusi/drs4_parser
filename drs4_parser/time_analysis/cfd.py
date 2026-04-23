import numpy as np
from lifetime_parser.energy_analysis.pulse_height import pulse_height_3pts_parabolic

def cfd_linear(t, v, fraction=0.3, window=1,
               pulse_height_method=pulse_height_3pts_parabolic):
    vmin = pulse_height_method(t, v)
    level = fraction * vmin

    idx = np.where(v <= level)[0]
    if len(idx) == 0:
        return np.nan

    i0 = idx[0]
    i_start = i0 - window
    i_stop  = i0 + window + 1

    valid = (i_start>1) & (i_stop < len(t)-1)
    if not valid:
        return np.nan
    # Fit v = a t + b
    coeff = np.polyfit(t[i_start:i_stop], v[i_start:i_stop], 1)
    # the 0 is in
    return (level - coeff[1]) / coeff[0]

