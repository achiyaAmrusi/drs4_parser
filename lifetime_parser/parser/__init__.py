"""
DRS4 binary file parser

File layout
-----------
  [File header]
    4 B   'DRS2'          file magic (byte 3 = version)
    4 B   'TIME'          time header marker
    4 B   'B#' + uint16   board serial number
    for each active channel:
      4 B   'C00N'         channel header
      1024 × float32       effective bin widths (ns)

  [Events, repeated until EOF]
    4 B   'EHDR'          event marker
    4 B   uint32          serial number (starts at 1)
    8 × uint16            year, month, day, hour, minute, second, millisecond, range_mv
    4 B   'B#' + uint16   board serial number
    4 B   'T#' + uint16   trigger cell
    for each active channel:
      4 B   'C00N'         channel header
      4 B   int32          scaler (Hz)
      1024 × uint16        voltage bins  (0 = RC-0.5V, 65535 = RC+0.5V)

Voltage:  V (mV) = (raw / 65535.0 - 0.5) * 1000 + range_mv

Time:     The header stores bin *widths* dt[0..1023].
          Each event has a trigger cell `tcell` that sets the start position.
              t[0] = 0
              t[i] = sum( dt[(tcell+j) % 1024]  for j in 0..i-1 )
          i.e. rotate widths by -tcell, cumsum, prepend 0.
"""

from .read import event_stream, read_first_n
from .properties import print_file_properties, file_properties

