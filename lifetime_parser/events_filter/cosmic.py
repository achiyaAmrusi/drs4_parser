import numpy as np

def event_cosmic_filter(
    waveforms: dict,
    channels=("CHN1", "CHN2"),
    thresholds=(-1.0, -1.0),
    peak_max_time_width=10.0,
    bin_time_size=0.2,
):
    """
    Returns True if event PASSES cosmic rejection (i.e. is NOT cosmic)
    waveforms[ch] = (time, voltage)
    """

    if len(channels) != len(thresholds):
        raise ValueError("channels and thresholds must have same length")

    delta_bins = int(peak_max_time_width / bin_time_size)

    for i, ch in enumerate(channels):

        if ch not in waveforms:
            return False  # missing channel → reject

        t, v = waveforms[ch]

        i_min = np.argmin(v)
        i_check = i_min + 2 * delta_bins

        if i_check >= v.size:
            continue  # cannot classify → treat as non-cosmic

        # cosmic condition
        if v[i_check] < thresholds[i]:
            return False  # cosmic

    return True