import numpy as np
from lifetime_parser.energy_analysis import baseline_estimation

def event_area_filter(
    event: dict,
    roi=(50, 400),                  # fixed ROI in samples
    approx_pulse_window=(20, 60),          # samples before/after minimum
    baseline_window=(0, 20),        # baseline samples
    normalization_percent=100.0,    # f in [%]
    lower_line=(0.0, 0.0),          # (a_low, b_low)
    upper_line=(0.0, 1.0),          # (a_high, b_high)
):
    """
    A filter which is based on an area ratio?
    the point here is to remove the pilup and cosmic events
    Returns True if event PASSES pile-up rejection

    Implements DDRS4PALS-style pulse-area filtering
    """

    f = normalization_percent / 100.0

    baseline = baseline_estimation(event, baseline_window)
    for ch in event:

        t, v = event[ch]
        dt = t[1] - t[0]

        # ---- baseline correction ----
        baseline = np.mean(v[baseline_window[0]:baseline_window[1]])
        v_corr = v - baseline

        # ---- pulse height ----
        i_min = np.argmin(v_corr)
        height = abs(v_corr[i_min])

        # ---- Apulse ----
        w_pre, w_post = approx_pulse_window
        i0 = max(i_min - w_pre, 0)
        i1 = min(i_min + w_post, v_corr.size)

        Apulse = np.sum(np.abs(v_corr[i0:i1])) * dt

        # ---- AROI ----
        r0, r1 = roi
        AROI = np.sum(np.abs(v_corr[r0:r1])) * dt

        if AROI <= 0:
            return False

        # ---- normalized area ----
        Aratio = Apulse / (f * AROI)

        # ---- bounding box ----
        a_low, b_low = lower_line
        a_high, b_high = upper_line

        lower = a_low * height + b_low
        upper = a_high * height + b_high

        if not (lower <= Aratio <= upper):
            return False  # pile-up detected

    return True
