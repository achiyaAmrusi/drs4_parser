from lifetime_parser.parser.read import event_stream
import xml.etree.ElementTree as ET
import numpy as np

def event_coincidence_filter(event, channels=("CHN1","CHN2"), thresholds=(-1.0,-1.0)):
    """
    Check if the event satisfies a simple coincidence:
    - Waveform in channel ch1 exceeds threshold1
    - Waveform in channel ch2 exceeds threshold2
    The threshold is negative because the signal is negative!
    Returns True if both conditions are satisfied.

    :param event: dict returned by event_stream
    :param channels: tuples of the channels names
    :param thresholds: thresholds voltage to eliminate events
    :return: bool
    """

    # helper to get minimal voltage of a channel
    def min_voltage(channel_tag):
        if channel_tag not in event:
            return None
        _, v = event[channel_tag]
        return np.min(v)

    # get minimum voltage for both channels
    v1 = min_voltage(channels[0])
    v2 = min_voltage(channels[1])

    # if either channel is missing, reject the event
    if v1 is None or v2 is None:
        return False

    # check if both exceed their thresholds
    return (v1 < thresholds[0]) and (v2 < thresholds[1])


def file_coincidence_parser(input_file, output_file, channels=("CHN1","CHN2"), thresholds=(-1.0,-1.0)):
    """
    Reads an input XML DRS4 file using event_stream,
    filters events using event_coincidence_filter,
    and writes passing events to a new minimal XML file.

    :param channels: tuples of the channels names
    :param thresholds: thresholds voltage to eliminate events
    Only includes the channels you filtered.
    """
    with open(output_file, "w", encoding="utf-8") as f_out:
        # XML header and root
        f_out.write('<?xml version="1.0" encoding="UTF-8"?>\n<DRSOSC>\n')

        for waveforms in event_stream(input_file):
            if event_coincidence_filter(waveforms, channels=channels, thresholds=thresholds):
                # minimal <Event>
                event_elem = ET.Element("Event")
                board_elem = ET.Element("Board_2580")

                for ch in channels:
                    if ch not in waveforms:
                        continue
                    t, v = waveforms[ch]
                    ch_elem = ET.Element(ch)
                    for ti, vi in zip(t, v):
                        data_elem = ET.Element("Data")
                        data_elem.text = f"{ti},{vi}"
                        ch_elem.append(data_elem)
                    board_elem.append(ch_elem)

                event_elem.append(board_elem)
                # write XML string
                f_out.write(ET.tostring(event_elem, encoding="unicode"))

        f_out.write("</DRSOSC>\n")