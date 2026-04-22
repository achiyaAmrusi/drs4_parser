__all__ = [
    "WAVEFORM_CH_NUM",
    "_MAGIC", "_TIME_HDR", "_EHDR",
    "_BOARD_TAG", "_TRIG_TAG", "_CHAN_TAGS",
    "_TIME_BYTES", "_VOLTAGE_BYTES",
]

WAVEFORM_CH_NUM = 1024

_MAGIC     = b"DRS2"
_TIME_HDR  = b"TIME"
_EHDR      = b"EHDR"
_BOARD_TAG = b"B#"
_TRIG_TAG  = b"T#"
_CHAN_TAGS  = {b"C001": "CH1", b"C002": "CH2", b"C003": "CH3", b"C004": "CH4"}

_TIME_BYTES    = WAVEFORM_CH_NUM * 4   # 1024 × float32 = 4096 B
_VOLTAGE_BYTES = WAVEFORM_CH_NUM * 2   # 1024 × uint16  = 2048 B