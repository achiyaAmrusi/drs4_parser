import numpy as np

def cfd_analog_time(
    t: np.ndarray,
    v: np.ndarray,
    fraction: float = 0.3,
    delay_time: float = 2.0
) -> float:
    """
    Analog-style CFD timing for a single waveform using time-based delay.

    Parameters
    ----------
    t : np.ndarray
        Time array (ns), strictly increasing.
    v : np.ndarray
        Voltage waveform (negative-going pulse).
    fraction : float
        Constant fraction (0 < fraction < 1).
    delay_time : float
        Delay time in ns.

    Returns
    -------
    float
        CFD time in ns, or NaN if no valid crossing is found.
    """
    if t.ndim != 1 or v.ndim != 1:
        raise ValueError("cfd_analog_time expects 1D arrays")

    if len(t) < 3 or delay_time <= 0:
        return np.nan

    # Interpolated delayed waveform
    v_delay = np.interp(t - delay_time, t, v, left=np.nan, right=np.nan)

    cfd = v_delay - fraction * v

    # Restrict to rising edge (before pulse minimum)
    i_peak = np.argmin(np.nan_to_num(cfd))
    valid = (~np.isnan(v_delay)) & (np.arange(len(v)) < i_peak)

    if np.count_nonzero(valid) < 2:
        return np.nan

    cfd_v = cfd[valid]
    t_v = t[valid]

    # Zero crossing: negative → positive
    idx = np.where((cfd_v[:-1] >= 0) & (cfd_v[1:] < 0))[0]
    if len(idx) == 0:
        return np.nan

    k = idx[-1]  # closest crossing to peak

    t1, t2 = t_v[k], t_v[k + 1]
    y1, y2 = cfd_v[k], cfd_v[k + 1]

    return t1 - y1 * (t2 - t1) / (y2 - y1)

def cfd_analog_time_batch(
    t: np.ndarray,
    v: np.ndarray,
    fraction: float = 0.3,
    delay_time: float = 2.0
) -> np.ndarray:
    """
    Batch wrapper for analog CFD timing.

    Parameters
    ----------
    t : np.ndarray
        Time array of shape (n_events, n_samples).
    v : np.ndarray
        Voltage array of shape (n_events, n_samples).
    fraction : float
        Constant fraction.
    delay_time : float
        Delay time in ns.

    Returns
    -------
    np.ndarray
        CFD times (ns) for each event. Invalid events return NaN.
    """
    if t.shape != v.shape:
        raise ValueError("t and v must have the same shape")

    n_events = t.shape[0]

    return np.fromiter(
        (cfd_analog_time(t[i], v[i], fraction, delay_time)
         for i in range(n_events)),
        dtype=float,
        count=n_events
    )
