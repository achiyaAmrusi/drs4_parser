import numpy as np

def baseline_estimation(
    event: dict,
    baseline_window=(0, 20),                  # ROI for baseline
):
    """
    Returns True if event PASSES pile-up rejection

    :param event: event from event_stream
    :param baseline_window: in ns
    Implements DDRS4PALS-style pulse-area filtering
    """

    baseline = {}
    for ch in event:
        t, v = event[ch]
        dt = t[1] - t[0]

        t_1_index = int(baseline_window[0]/dt)
        t_2_index = int(baseline_window[1] / dt)
        # ---- baseline correction ----
        baseline.update({ch:np.mean(v[t_1_index:t_2_index])})

    return baseline