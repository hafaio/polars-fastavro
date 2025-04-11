from contextlib import ExitStack
from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias

import fastavro
import polars as pl

AvroSchema: TypeAlias = str | list["AvroSchema"] | dict[str, "AvroSchema"]


@dataclass(frozen=True)
class DataTypeFormatter:
    promote_ints: bool
    promote_array: bool

    def format_dtype(self, dtype: pl.DataType) -> AvroSchema:  # noqa: PLR0911, PLR0912, PLR0915
        formatted: AvroSchema
        match dtype:
            case pl.Null:
                return "null"
            case pl.Boolean:
                formatted = "boolean"
            case pl.Int32:
                formatted = "int"
            case pl.Int8 | pl.Int16 | pl.UInt8 | pl.UInt16 if self.promote_ints:
                formatted = "int"
            case pl.Date:
                formatted = {"type": "int", "logicalType": "date"}
            case pl.Datetime:
                match (dtype.time_unit, dtype.time_zone):
                    case ("ms", "UTC"):
                        formatted = {"type": "long", "logicalType": "timestamp-millis"}
                    case ("us", "UTC"):
                        formatted = {"type": "long", "logicalType": "timestamp-micros"}
                    case ("ms", None):
                        formatted = {
                            "type": "long",
                            "logicalType": "local-timestamp-millis",
                        }
                    case ("us", None):
                        formatted = {
                            "type": "long",
                            "logicalType": "local-timestamp-micros",
                        }
                    case _:
                        raise ValueError(f"unsupported dtype: {dtype}")
            case pl.Int64:
                formatted = "long"
            case pl.UInt32 if self.promote_ints:
                formatted = "long"
            case pl.Float32:
                formatted = "float"
            case pl.Float64:
                formatted = "double"
            case pl.Enum:
                formatted = {
                    "type": "enum",
                    "name": "fastavro_enum",
                    "symbols": dtype.categories.to_list(),
                }
            case pl.Binary:
                formatted = "bytes"
            case pl.String:
                formatted = "string"
            case pl.List:
                formatted = {
                    "type": "array",
                    "items": self.format_dtype(dtype.inner),  # type: ignore
                }
            case pl.Array if self.promote_array:
                formatted = {
                    "type": "array",
                    "items": self.format_dtype(dtype.inner),  # type: ignore
                }
            case pl.Struct:
                fields: list[AvroSchema] = []
                for field in dtype.fields:
                    fields.append(
                        {"name": field.name, "type": self.format_dtype(field.dtype)}  # type: ignore
                    )
                formatted = {
                    "type": "record",
                    "name": "fastavro_record",
                    "fields": fields,
                }
            case _:
                # NOTE time is hard to truncate because we'd have to match the
                # schema ahead of time
                # NOTE datetime is similarly difficult because of time zone
                raise ValueError(f"unsupported dtype: {dtype}")
        return ["null", formatted]

    def format_schema(self, schema: pl.Schema) -> AvroSchema:
        fields: list[AvroSchema] = []
        for name, dtype in schema.items():
            fields.append({"name": name, "type": self.format_dtype(dtype)})
        return {"type": "record", "name": "fastavro_schema", "fields": fields}


def write_avro(  # noqa: PLR0913
    frame: pl.DataFrame,
    dest: str | Path | BinaryIO,
    *,
    batch_size: int | None = None,
    promote_ints: bool = True,
    promote_array: bool = True,
    codec: Literal["null", "deflate", "snappy"] = "null",
) -> None:
    """Write a DataFrame as an avro file.

    Parameters
    ----------
    frame : The DataFrame to write.
    dest : Where to write the frame to.
    promote_ints : Whether to promote ints to a large size that avro supports.
    promote_arrays : Whether to write Arrays as Lists.
    coded : Codec for dest.
    """
    schema = DataTypeFormatter(
        promote_ints=promote_ints,
        promote_array=promote_array,
    ).format_schema(frame.schema)
    with ExitStack() as stack:
        match dest:
            case str() | Path():
                fo = stack.enter_context(open(path.expanduser(dest), "wb"))
            case _:
                fo = dest
        if batch_size is None:
            frames = [frame]
        else:
            frames = [
                frame[i : i + batch_size].rechunk()
                for i in range(0, len(frame), batch_size)
            ]
        fastavro.writer(
            fo,
            schema,
            (row for frame in frames for row in frame.iter_rows(named=True)),
            codec=codec,
        )
