r"""Polars io-plugin backed by fastavro.

This plugin allows reading, writing, and scanning avro files into polars
DataFrames using the fastavro library.

Usage
-----

 .. code-block:: python

    from polars_fastavro import scan_avro, read_avro, write_avro

    frame = scan_avro(...).collect()  # or `read_avro(...)`
    write_avro(frame, dest)

Limitations
-----------

1. Because it uses python types an an intermediary, it's slow, (30x read to 80x
   write).
2. Since this is ultimately converting between avro and arrow, it has no support
   for avro maps, unions (other than null), names for certain types
3. Every type is treated as as nullable.
4. Additionally, some types could in theory be supported by aren't for technical
   reasons. These include fixed, decimal, uuid, time, and
   duration.
5. Timestamp support is limited. local-timestamp-\*s are treated as Datetime
   without tz info, while timestamp-\*s are reated as UTC Datetime. Writing
   Datetimes with nano-precision is also not supported.
6. This can't read cloud files, as that functionality isn't exposed in python to
   my knowledge.
"""

from ._scan import read_avro, scan_avro
from ._sink import write_avro

__all__ = ("scan_avro", "read_avro", "write_avro")
