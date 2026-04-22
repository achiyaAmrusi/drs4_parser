import numpy as np
from typing import Callable, Dict


def pulse_height_fit(
    t: np.ndarray,
    v: np.ndarray,
    window: int = 2
) -> float:
    """
    Estimate pulse height using a quadratic fit around the waveform minimum.

    A second-degree polynomial is fitted to a local window centered at the
    minimum sample. The interpolated minimum of the parabola is returned.

    Parameters
    ----------
    t : np.ndarray
        Time array (same length as `v`).
    v : np.ndarray
        Voltage waveform (negative-going pulse expected).
    window : int, optional
        Half-width of the fitting window in samples.
        Total points used = 2*window + 1.

    Returns
    -------
    float
        Estimated pulse minimum (voltage).
        Returns NaN if the minimum is too close to the edges or the fit fails.
    """
    ind_min = np.argmin(v)
    valid = (ind_min >= window) and (ind_min < len(v) - window)

    if not valid:
        return np.nan

    idx = ind_min + np.arange(-window, window + 1)

    try:
        a, b, c = np.polyfit(t[idx], v[idx], deg=2)
    except np.linalg.LinAlgError:
        return np.nan

    # Require downward opening parabola
    if a <= 0:
        return np.nan

    # Parabolic minimum: -b^2 / (4a) + c
    return -b**2 / (4 * a) + c


def pulse_height_3pts_parabolic(
    t: np.ndarray,
    v: np.ndarray,
    window: int | None = None
) -> float:
    """
    Estimate pulse height using 3-point parabolic interpolation with actual times.

    Uses the minimum sample and its immediate neighbors to fit a parabola
    and find the sub-sample voltage minimum. Works for non-uniformly spaced
    time samples (e.g., DRS4 calibrated times).

    Parameters
    ----------
    t : np.ndarray
        Time array for the waveform (ns). Must be same length as `v`.
    v : np.ndarray
        Voltage waveform (negative-going pulse expected).
    window : None
        Ignored. Present for API compatibility.

    Returns
    -------
    float
        Estimated pulse minimum (voltage). Returns NaN if the minimum is at
        the edge of the array or if the parabola curvature is invalid.
    """
    ind_min = np.argmin(v)

    # Cannot fit if minimum is at the first or last sample
    if ind_min == 0 or ind_min == len(v) - 1:
        return np.nan

    # Take three points: minimum and its immediate neighbors
    t_window = t[ind_min - 1:ind_min + 2]
    y_window = v[ind_min - 1:ind_min + 2]

    # Solve for parabola coefficients: v(t) = a t^2 + b t + c
    A = np.vstack([t_window**2, t_window, np.ones(3)]).T
    try:
        a, b, c = np.linalg.solve(A, y_window)
    except np.linalg.LinAlgError:
        return np.nan

    # Require positive curvature for a true minimum
    if a <= 0:
        return np.nan

    # Parabolic minimum: V_min = -b^2 / (4a) + c
    V_min = -b**2 / (4*a) + c
    return V_min


import numpy as np


def pulse_height_robust(
        t: np.ndarray,
        v: np.ndarray,
        window: int | None = None
) -> np.ndarray:
    """
    Fully vectorized 3-point parabolic pulse-height estimator for many events
    using actual time arrays (supports non-uniform sampling).

    Computes pulse heights for a batch of waveforms by fitting a parabola
    through the minimum sample and its immediate neighbors for each event.

    Parameters
    ----------
    t : np.ndarray
        Time array of shape (n_events, n_samples), in ns.
    v : np.ndarray
        Voltage array of shape (n_events, n_samples), negative-going pulses.
    window : None
        Ignored. Present for API compatibility.

    Returns
    -------
    np.ndarray
        Array of pulse heights for each event.
        Events with the minimum at the edge or invalid curvature return NaN.
    """
    n_events, n_points = v.shape
    idx_min = np.argmin(v, axis=1)

    # Only events where minimum is not at the edges
    valid = (idx_min > 0) & (idx_min < n_points - 1)
    if not np.any(valid):
        return np.full(n_events, np.nan)

    idx_min_valid = idx_min[valid]
    v_valid = v[valid]
    t_valid = t[valid]

    # Extract 3-point windows
    idx_window = idx_min_valid[:, None] + np.array([-1, 0, 1])[None, :]
    v_win = np.take_along_axis(v_valid, idx_window, axis=1)
    t_win = np.take_along_axis(t_valid, idx_window, axis=1)

    # Solve parabola coefficients using Cramer's rule (vectorized)
    t1, t2, t3 = t_win[:, 0], t_win[:, 1], t_win[:, 2]
    y1, y2, y3 = v_win[:, 0], v_win[:, 1], v_win[:, 2]

    # Determinant of the coefficient matrix
    detA = (t1 - t2) * (t1 - t3) * (t2 - t3)

    # Avoid divide by zero
    mask_det = detA != 0

    a = np.full(len(idx_min_valid), np.nan)
    b = np.full(len(idx_min_valid), np.nan)
    c = np.full(len(idx_min_valid), np.nan)

    # Only solve for events with non-singular system
    if np.any(mask_det):
        # Using Cramer's rule
        a[mask_det] = (y1[mask_det] * (t2[mask_det] - t3[mask_det]) +
                       y2[mask_det] * (t3[mask_det] - t1[mask_det]) +
                       y3[mask_det] * (t1[mask_det] - t2[mask_det])) / detA[mask_det]

        b[mask_det] = (y1[mask_det] * (t3[mask_det] ** 2 - t2[mask_det] ** 2) +
                       y2[mask_det] * (t1[mask_det] ** 2 - t3[mask_det] ** 2) +
                       y3[mask_det] * (t2[mask_det] ** 2 - t1[mask_det] ** 2)) / detA[mask_det]

        c[mask_det] = (y1[mask_det] * (t2[mask_det] * t3[mask_det] * (t2[mask_det] - t3[mask_det])) +
                       y2[mask_det] * (t3[mask_det] * t1[mask_det] * (t3[mask_det] - t1[mask_det])) +
                       y3[mask_det] * (t1[mask_det] * t2[mask_det] * (t1[mask_det] - t2[mask_det]))) / detA[mask_det]

    # Only keep events with upward-opening parabola (a > 0)
    mask_valid = a > 0
    height = np.full(len(idx_min_valid), np.nan)
    height[mask_valid] = -b[mask_valid] ** 2 / (4 * a[mask_valid]) + c[mask_valid]

    # Put back into full array aligned with original events
    height_full = np.full(n_events, np.nan)
    height_full[valid] = height
    return height_full


def event_pulse_height(
    event: Dict,
    window: int = 1,
    method: Callable = pulse_height_3pts_parabolic
) -> Dict:
    """
    Compute pulse heights for all channels in a single event.

    Parameters
    ----------
    event : dict
        Dictionary mapping channel -> (time_array, voltage_array),
        as returned by `event_stream`.
    window : int, optional
        Window parameter forwarded to the pulse-height method.
    method : callable, optional
        Pulse-height extraction function with signature
        `(t, v, window) -> float`.

    Returns
    -------
    dict
        Dictionary mapping channel -> pulse height (float).
    """
    return {
        ch: method(t, v, window)
        for ch, (t, v) in event.items()
    }
