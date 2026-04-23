from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from lifetime_parser.parser.constants import *

@dataclass
class _FileHeader:
    board_serial: int                     # uint16
    bin_widths: dict[str, np.ndarray]     # channel → float32 (1024,)
    header_bytes: int                     # file offset where events begin
    channels: list[str]                   # ordered list of active channels

@dataclass
class DRS4FileProperties:
    board_serial: int
    channels: list[str]     # e.g. ['CH1', 'CH2']
    n_events: int
    samples_per_event: int  # always 1024
    header_bytes: int       # byte offset where events start
    event_bytes: int        # bytes per complete event
    file_bytes: int

    def summary(self) -> None:
        print(
            f"DRS4 binary\n"
            f"  board serial    : {self.board_serial}\n"
            f"  channels        : {len(self.channels)} ({', '.join(self.channels)})\n"
            f"  samples / event : {self.samples_per_event}\n"
            f"  header bytes    : {self.header_bytes}\n"
            f"  events          : {self.n_events}\n"
            f"  event size      : {self.event_bytes / 1024:.1f} KB\n"
            f"  file size       : {self.file_bytes / 1024 / 1024:.2f} MB"
        )


def _read_file_header(f) -> _FileHeader:
    """
    Parse the calibration section at the top of the file.
    Leaves the file pointer at the first 'EHDR'.
    """
    magic = f.read(4)
    if magic[:3] != b"DRS":
        raise ValueError(f"Not a DRS4 file — expected 'DRS2', got {magic!r}")

    time_hdr = f.read(4)
    if time_hdr != _TIME_HDR:
        raise ValueError(f"Expected 'TIME' header, got {time_hdr!r}")

    board_tag = f.read(4)
    if board_tag[:2] != _BOARD_TAG:
        raise ValueError(f"Expected 'B#' board tag, got {board_tag!r}")
    board_serial = struct.unpack("<H", board_tag[2:])[0]

    bin_widths: dict[str, np.ndarray] = {}
    channels: list[str] = []

    while True:
        tag = f.read(4)
        if not tag or len(tag) < 4:
            break
        if tag == _EHDR:
            f.seek(-4, 1)
            break
        if tag in _CHAN_TAGS:
            ch = _CHAN_TAGS[tag]
            bin_widths[ch] = np.frombuffer(f.read(_TIME_BYTES), dtype="<f4").copy()
            channels.append(ch)
        else:
            raise ValueError(
                f"Unexpected tag {tag!r} in file header at offset {f.tell()-4:#x}"
            )

    return _FileHeader(
        board_serial = board_serial,
        bin_widths   = bin_widths,
        header_bytes = f.tell(),
        channels     = channels,
    )


def file_properties(drs4_file: str | Path) -> DRS4FileProperties:
    """
    Read metadata from a DRS4 binary file.
    Event count is O(1) — derived from file size once header size is known.
    """
    path = Path(drs4_file)

    with open(path, "rb") as f:
        hdr = _read_file_header(f)

    n_ch = len(hdr.channels)

    # bytes per event:
    #   4  EHDR
    #   4  serial uint32
    #   16 datetime + range (8 × uint16)
    #   4  B# + board serial
    #   4  T# + trigger cell
    #   per channel: 4 (C00N) + 4 (scaler int32) + 2048 (voltage)
    event_bytes = (
        4 + 4 + 16 + 4 + 4
        + n_ch * (4 + 4 + _VOLTAGE_BYTES)
    )

    file_bytes = path.stat().st_size
    n, remainder = divmod(file_bytes - hdr.header_bytes, event_bytes)
    if remainder != 0:
        warnings.warn(
            f"File size remainder = {remainder} B — file may be truncated. "
            f"Reporting {n} complete events."
        )

    return DRS4FileProperties(
        board_serial      = hdr.board_serial,
        channels          = hdr.channels,
        n_events          = n,
        samples_per_event = WAVEFORM_CH_NUM,
        header_bytes      = hdr.header_bytes,
        event_bytes       = event_bytes,
        file_bytes        = file_bytes,
    )


def print_file_properties(drs4_file: str | Path) -> None:
    file_properties(drs4_file).summary()

