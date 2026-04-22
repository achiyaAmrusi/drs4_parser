from __future__ import annotations
import struct
from pathlib import Path
from typing import Iterator
import numpy as np
import xarray as xr

from lifetime_parser.parser.properties import _read_file_header, file_properties
from lifetime_parser.parser.constants import *


# --- time reconstruction -------

def _build_time(bin_widths: np.ndarray, tcell: int) -> np.ndarray:
    """
    Reconstruct the time axis for one channel given the trigger cell.

    Parameters
    ----------
    bin_widths : float32 array, shape (1024,) — effective bin widths in ns
    tcell      : trigger cell for this event

    Returns
    -------
    float32 array, shape (1024,), t[0] = 0
    """
    rotated = np.roll(bin_widths, -tcell)
    t = np.empty(WAVEFORM_CH_NUM, dtype=np.float32)
    t[0] = 0.0
    t[1:] = np.cumsum(rotated[:-1])
    return t


def _cell0_time(bin_widths: np.ndarray, tcell: int) -> float:
    """Absolute time of cell #0 — used for cross-channel alignment."""
    rotated = np.roll(bin_widths, -tcell)
    idx = (WAVEFORM_CH_NUM - tcell) % WAVEFORM_CH_NUM
    return float(np.sum(rotated[:idx]))

# --- event stream -------

def event_stream(
    drs4_file: str | Path,
    *,
    voltage_in_mv: bool = True,
    align_channels: bool = False,
) -> Iterator[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Yield one event at a time from a DRS4 binary file.

    Each yielded value maps channel name → (time, voltage):
        time    : float32 array, length 1024, nanoseconds (t[0] = 0)
        voltage : float32 array, length 1024, millivolts  (if voltage_in_mv=True)
                  or raw uint16                            (if voltage_in_mv=False)

    Parameters
    ----------
    drs4_file      : str or Path
    voltage_in_mv  : convert raw ADC to mV (default True)
    align_channels : shift time arrays so cell #0 is aligned across channels
                     (recommended for timing measurements, default False)
    """
    path = Path(drs4_file)

    with open(path, "rb") as f:
        hdr = _read_file_header(f)

        while True:
            # ── event header ─────────────────────────────────────────────
            tag = f.read(4)
            if not tag:
                return
            if tag != _EHDR:
                raise ValueError(
                    f"Expected 'EHDR' at offset {f.tell()-4:#x}, got {tag!r}"
                )

            _serial = struct.unpack("<I", f.read(4))[0]
            _year, _month, _day, _hour, _minute, _second, _millisec, range_mv = \
                struct.unpack("<8H", f.read(16))

            # skip B# board tag (4 B) — single board, we already know the serial
            f.seek(4, 1)

            tcell = struct.unpack("<H", f.read(4)[2:])[0]   # 'T#' + uint16

            # ── channels ─────────────────────────────────────────────────
            event: dict[str, tuple[np.ndarray, np.ndarray]] = {}

            for _ in hdr.channels:
                ch_tag = f.read(4)
                if ch_tag not in _CHAN_TAGS:
                    raise ValueError(
                        f"Expected channel tag at {f.tell()-4:#x}, got {ch_tag!r}"
                    )
                ch = _CHAN_TAGS[ch_tag]

                f.seek(4, 1)   # skip scaler
                raw = np.frombuffer(f.read(_VOLTAGE_BYTES), dtype="<u2").copy()

                t = _build_time(hdr.bin_widths[ch], tcell)

                if voltage_in_mv:
                    v = (raw.astype(np.float32) /np.iinfo(np.uint16).max - 0.5) * 1000.0 \
                        + float(range_mv)
                else:
                    v = raw

                event[ch] = (t, v)

            # ── optional cross-channel time alignment ─────────────────────
            if align_channels and len(event) > 1:
                ref_ch = hdr.channels[0]
                t0_ref = _cell0_time(hdr.bin_widths[ref_ch], tcell)
                event = {
                    ch: (
                        t - np.float32(_cell0_time(hdr.bin_widths[ch], tcell) - t0_ref),
                        v,
                    )
                    for ch, (t, v) in event.items()
                }

            yield event


# ── bulk reader ───────────────────────────────────────────────────────────────

def read_first_n(
    drs4_file: str | Path,
    n: int,
    *,
    channels: list[str] | None = None,
    voltage_in_mv: bool = True,
    align_channels: bool = False,
) -> xr.Dataset:
    """
    Read the first N complete events and return an xarray Dataset.

    Parameters
    ----------
    drs4_file      : str or Path
    n              : number of events to read
    channels       : subset of channels to keep; None = all channels in file
    voltage_in_mv  : convert raw ADC to mV (default True)
    align_channels : apply cross-channel time alignment (default False)

    Returns
    -------
    xr.Dataset
        data vars:
            voltage  (channel, event, point)  – float32, mV
            time     (channel, event, point)  – float32, ns
        coords:
            channel  – e.g. ['CH1', 'CH2']
            event    – 0 … count-1
            point    – 0 … 1023
    """
    props = file_properties(drs4_file)
    want  = channels if channels is not None else props.channels

    unknown = set(want) - set(props.channels)
    if unknown:
        raise ValueError(
            f"Channels not found in file: {unknown}. Available: {props.channels}"
        )

    n_ch    = len(want)
    voltage = np.empty((n_ch, n, WAVEFORM_CH_NUM), dtype=np.float32)
    time    = np.empty((n_ch, n, WAVEFORM_CH_NUM), dtype=np.float32)

    count = 0
    for waveforms in event_stream(
        drs4_file, voltage_in_mv=voltage_in_mv, align_channels=align_channels
    ):
        if not all(ch in waveforms for ch in want):
            continue

        for ich, ch in enumerate(want):
            t, v = waveforms[ch]
            time[ich, count, :]    = t
            voltage[ich, count, :] = v

        count += 1
        if count >= n:
            break

    voltage = voltage[:, :count, :]
    time    = time[:,    :count, :]

    return xr.Dataset(
        data_vars={
            "voltage": (("channel", "event", "point"), voltage),
            "time":    (("channel", "event", "point"), time),
        },
        coords={
            "channel": list(want),
            "event":   np.arange(count),
            "point":   np.arange(WAVEFORM_CH_NUM),
        },
        attrs={
            "source":               str(drs4_file),
            "samples_per_waveform": WAVEFORM_CH_NUM,
        },
    )
