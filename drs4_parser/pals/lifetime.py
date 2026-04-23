import numpy as np
import xarray as xr

from lifetime_parser.parser import event_stream
from lifetime_parser.events_filter import event_coincidence_filter
from lifetime_parser.time_analysis import cfd_linear, cfd_analog_time

def good_event(event, channels=("CHN1","CHN2"), thresholds=(-1.0, -1.0)):
    """
    filter for the events for pals spectroscopy.
    I should add more such filter (for example cosmic filter)

    :param event: dict returned by event_stream
    :param channels: tuples of the channels names
    :param threshold1, threshold2: threshold voltage to eliminate events
    :return: bool
    """
    isgood =event_coincidence_filter(event, channels=channels, thresholds=thresholds)
    if not isgood:
        return False
    return True


def read_lifetime(
    input_file,
    channels=("CHN1", "CHN2"),
    thresholds=(-1.0, -1.0),
    cfd_method="linear",      # "analog" | "linear"
    cfd_fraction=0.25,
    cfd_delay=None,
    max_num_events=100000):
    """
    Read waveform data from .lftmbin file and extract pals timing using
    Constant Fraction Discrimination (CFD).

    For each valid event, the function:
      1. Reads waveforms from two channels
      2. Applies amplitude thresholds to reject noise events
      3. Extracts pulse height (minimum voltage, assuming negative pulses)
      4. Determines the pulse time using either linear or analog CFD
      5. Stores results in an xarray Dataset

    Parameters
    ----------
    input_file : str or Path
        Input lftmbin file containing waveform data readable by `event_stream`.
        to create one from DRS4 xml files use convert_xml_to_bin from lifetime_parser.converter

    channels : tuple of str, optional
        Names of the two channels to be used for the pals measurement.
        Default is ("CHN1", "CHN2").

    thresholds : tuple of float, optional
        Voltage thresholds for each channel. Events with pulse minima
        above these values are rejected. Default is (-1.0, -1.0).

    cfd_method : {"analog", "linear"}, optional
        CFD algorithm used to determine the pulse timing:
          - "linear": Linear interpolation at a fixed fraction of pulse height.
          - "analog": Analog CFD using a delayed and scaled copy of the signal.
        Default is "analog".

    cfd_fraction : float, optional
        Constant fraction of the pulse height used for timing (0 < fraction < 1).
        For linear CFD, this is the fraction of the pulse minimum.
        For analog CFD, this is the attenuation factor applied to the original
        signal before subtraction. Default is 0.25.

    cfd_delay : float, optional
        Delay time (in the same units as the time array) used for analog CFD.
        Must be provided when `cfd_method="analog"`. Ignored for linear CFD.

    max_num_events : int, optional
        Maximum number of valid events to process. Default is 100000.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing per-event quantities with dimension ``event``:

        - ``energy_1``, ``energy_2`` :
            Pulse heights (minimum voltage) for each channel.
        - ``time_1``, ``time_2`` :
            CFD timing for each channel.

        Dataset attributes store the CFD configuration:
        ``cfd_method``, ``cfd_fraction``, and ``cfd_delay``.

    Notes
    -----
    - Pulses are assumed to be negative-going.
    - Timing failures are stored as NaN values.
    - Analog CFD timing may exhibit reduced time walk but increased
      sensitivity to noise and baseline fluctuations compared to
      linear CFD.
    - The function does not compute the pals directly; it provides
      per-channel timestamps to allow flexible post-processing.

    See Also
    --------
    cfd_linear : Linear constant fraction discriminator.
    cfd_analog_time : Analog (delayed) constant fraction discriminator.
    """
    ch1_time = np.full(max_num_events, np.nan)
    ch2_time = np.full(max_num_events, np.nan)
    ch1_height = np.zeros(max_num_events)
    ch2_height = np.zeros(max_num_events)

    event_num = 0

    for waveforms in event_stream(input_file):

        if event_num >= max_num_events:
            break

        if not good_event(waveforms, channels=channels, thresholds=thresholds):
            continue

        time_1, voltage_1 = waveforms[channels[0]]
        time_2, voltage_2 = waveforms[channels[1]]

        # pulse height (negative pulses)
        ch1_height[event_num] = voltage_1.min()
        ch2_height[event_num] = voltage_2.min()

        # --- CFD timing ---
        if cfd_method == "linear":
            ch1_time[event_num] = cfd_linear(
                time_1, voltage_1, fraction=cfd_fraction
            )
            ch2_time[event_num] = cfd_linear(
                time_2, voltage_2, fraction=cfd_fraction
            )

        elif cfd_method == "analog":
            if cfd_delay is None:
                raise ValueError("cfd_delay must be provided for analog CFD")

            ch1_time[event_num] = cfd_analog_time(
                time_1, voltage_1,
                fraction=cfd_fraction,
                delay_time=cfd_delay,
            )
            ch2_time[event_num] = cfd_analog_time(
                time_2, voltage_2,
                fraction=cfd_fraction,
                delay_time=cfd_delay,
            )

        else:
            raise ValueError(f"Unknown cfd_method: {cfd_method}")

        event_num += 1

    # resize arrays
    ch1_time = ch1_time[:event_num]
    ch2_time = ch2_time[:event_num]
    ch1_height = ch1_height[:event_num]
    ch2_height = ch2_height[:event_num]

    # build Dataset
    ds = xr.Dataset(
        data_vars={
            "energy_1": ("event", ch1_height),
            "energy_2": ("event", ch2_height),
            "time_1": ("event", ch1_time),
            "time_2": ("event", ch2_time),
        },
        coords={
            "event": np.arange(event_num),
        },
        attrs={
            "cfd_method": cfd_method,
            "cfd_fraction": cfd_fraction,
            "cfd_delay": cfd_delay,
        },
    )

    return ds
