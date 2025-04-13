import glob as libglob
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import BinaryIO, TypeVar

import fastavro
import polars as pl
from polars.io.plugins import register_io_source

R = TypeVar("R")


def unwrap_nullable(schema: object) -> object:
    match schema:
        case (
            ["null", object() as other]
            | [object() as other, "null"]
            | [object() as other]
        ):
            return other
        case _:
            return schema


@dataclass(frozen=True)
class DataTypeParser:
    parse_logical_types: bool
    single_col_name: str | None

    def parse_dtype(self, dtype: object) -> pl.DataType:  # noqa: PLR0911, PLR0912
        match unwrap_nullable(dtype):
            case {"type": "long", "logicalType": "timestamp-millis"}:
                return pl.Datetime("ms", "UTC")
            case {"type": "long", "logicalType": "timestamp-micros"}:
                return pl.Datetime("us", "UTC")
            case {"type": "long", "logicalType": "timestamp-nanos"}:
                return pl.Datetime("ns", "UTC")
            case {"type": "long", "logicalType": "local-timestamp-millis"}:
                return pl.Datetime("ms", None)
            case {"type": "long", "logicalType": "local-timestamp-micros"}:
                return pl.Datetime("us", None)
            case {"type": "long", "logicalType": "local-timestamp-nanos"}:
                return pl.Datetime("ns", None)
            case {"type": "int", "logicalType": "date"}:
                return pl.Date()
            case {
                "type": "int" | "long" | "bytes" | "string",
                "logicalType": str(),
            } if not self.parse_logical_types:
                raise ValueError(f"tried to parse {dtype} without logical-type parsing")
            case "null":
                return pl.Null()
            case "boolean":
                return pl.Boolean()
            case "int" | {"type": "int"}:
                return pl.Int32()
            case "long" | {"type": "long"}:
                return pl.Int64()
            case "float":
                return pl.Float32()
            case "double":
                return pl.Float64()
            case {
                "type": "enum",
                "symbols": [*_] as symbs,  # type: ignore
            }:
                return pl.Enum(symbs)  # type: ignore
            case "bytes" | {"type": "bytes"}:
                return pl.Binary()
            case "string" | {"type": "string"}:
                return pl.String()
            case {"type": "array", "items": object() as inner}:
                return pl.List(self.parse_dtype(inner))
            case {"type": "record", "fields": [*_] as fields}:  # type: ignore
                parsed: list[tuple[str, pl.DataType]] = []
                for field in fields:  # type: ignore
                    match field:
                        case {"name": str() as name, "type": object() as dtype}:
                            parsed.append((name, self.parse_dtype(dtype)))
                        case _:  # pragma: no cover
                            raise RuntimeError(f"invalid field definition: {field}")
                return pl.Struct(dict(parsed))
            case unwrapped:
                raise NotImplementedError(f"unhandled datatype: {unwrapped}")

    def parse_schema(self, schema: object) -> tuple[pl.Schema, bool]:
        match schema:
            case {"type": "record", "fields": [*_] as fields}:  # type: ignore
                parsed: list[tuple[str, pl.DataType]] = []
                for field in fields:  # type: ignore
                    match field:
                        case {"name": str() as name, "type": object() as dtype}:
                            parsed.append((name, self.parse_dtype(dtype)))
                        case _:  # pragma: no cover
                            raise RuntimeError(f"invalid field definition: {field}")
                return pl.Schema(parsed), False
            case _ if self.single_col_name is not None:
                col_dtype = self.parse_dtype(schema)
                return pl.Schema([(self.single_col_name, col_dtype)]), True
            case _:
                raise NotImplementedError(
                    f"top-level schema must be a record schema: {schema}"
                )


def open_sources(
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    glob: bool,
) -> Iterator[BinaryIO]:
    match sources:
        case [*_]:
            normed = sources
        case _:
            normed = [sources]

    for source in normed:
        match source:
            case str() | Path():
                expanded = path.expanduser(source)
                # sort for deterministic ordering of files
                globbed = sorted(libglob.glob(expanded)) if glob else [expanded]
                for fpath in globbed:
                    with open(fpath, "rb") as fo:
                        yield fo
            case _:
                yield source


def iter_readers(
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    *,
    glob: bool,
    parse_logical_types: bool,
    single_col_name: str | None,
) -> tuple[pl.Schema, bool, Iterator[Iterable[object]]]:
    source_iter = open_sources(sources, glob)
    if (first := next(source_iter, None)) is None:
        raise ValueError("sources were empty")

    parser = DataTypeParser(
        parse_logical_types=parse_logical_types, single_col_name=single_col_name
    )

    reader = fastavro.reader(first)
    schema, singleton = parser.parse_schema(reader.writer_schema)

    def rest() -> Iterator[Iterable[object]]:
        yield reader
        for i, fo in enumerate(source_iter, 1):
            next_reader = fastavro.reader(fo)
            new_schema, new_single = parser.parse_schema(next_reader.writer_schema)
            if new_schema == schema and singleton == new_single:
                yield next_reader
            else:
                raise RuntimeError(
                    f"schema of source {i:d} didn't match schema of source 0\n{next_reader.writer_schema} != {schema}"
                )

    return schema, singleton, rest()


def chunk(it: Iterable[R], chunk_size: int) -> Iterator[list[R]]:
    assert chunk_size > 0
    chunk: list[R] = []
    for val in it:
        chunk.append(val)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def scan_avro(
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    *,
    convert_logical_types: bool = False,
    batch_size: int = 32768,
    glob: bool = True,
    single_col_name: str | None = None,
) -> pl.LazyFrame:
    """Scan Avro files.

    Parameters
    ----------
    sources : The source(s) to scan.
    convert_logical_types : If true, logical types that can't be parsed, but are
        backed by physical types that can will be parsed as those physical types
        instead.
    batch_size : How many rows to attempt to read at a time.
    glob : If true, expand path sources with glob patterns.
    single_col_name : If not None and the avro schema isn't a record, wrap
        values in a record with a single field called `single_col_name`.
    """
    def_batch_size = batch_size

    schema: pl.Schema | None = None
    singleton: bool | None = None
    opened_readers: Iterator[Iterable[object]] | None = None

    def get_schema() -> pl.Schema:
        nonlocal schema, singleton, opened_readers
        if schema is not None:
            return schema
        elif opened_readers is not None:
            raise RuntimeError(  # pragma: no cover
                "internal error: schema not set but readers is"
            )
        else:
            schema, singleton, readers = iter_readers(
                sources,
                glob=glob,
                parse_logical_types=convert_logical_types,
                single_col_name=single_col_name,
            )
            opened_readers = readers
            return schema

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        if schema is None or singleton is None:  # pragma: no cover
            raise RuntimeError(
                "internal error: schema not defined when it needed to be"
            )
        nonlocal opened_readers
        if opened_readers is None:
            _, _, readers = iter_readers(
                sources,
                glob=glob,
                parse_logical_types=convert_logical_types,
                single_col_name=single_col_name,
            )
        else:
            readers = opened_readers
            opened_readers = None

        # if we parsed a singleton schema, then wrap to make them records
        records = (rec for reader in readers for rec in reader)
        if singleton:
            records = ({single_col_name: rec} for rec in records)

        for batch in chunk(records, batch_size or def_batch_size):
            lazy = pl.from_dicts(batch, schema).lazy()  # type: ignore
            if with_columns is not None:
                lazy = lazy.select(with_columns)
            if predicate is not None:
                lazy = lazy.filter(predicate)  # type: ignore
            frame = lazy.collect()
            if n_rows is None:
                yield frame
            else:
                frame = frame[:n_rows]
                n_rows -= len(frame)
                yield frame
                if n_rows == 0:
                    break

    return register_io_source(source_generator, schema=get_schema)


def read_avro(  # noqa: PLR0913
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    *,
    columns: Sequence[int | str] | None = None,
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    rechunk: bool = False,
    convert_logical_types: bool = False,
    batch_size: int = 32768,
    glob: bool = True,
    single_col_name: str | None = None,
) -> pl.DataFrame:
    """Read an avro file.

    Parameters
    ----------
    sources : The source(s) to read.
    columns : Columns to select from the read sources.
    n_rows : The maximum number of rows to read.
    row_index_name : If not None, the name to assign as a row index.
    row_index_offset : Where to position the row index column.
    rechunk : Whether to rechunk the frame so it's contiguous.
    convert_logical_types : If true, logical types that can't be parsed, but are
        backed by physical types that can will be parsed as those physical types
        instead.
    batch_size : How many rows to read at a time.
    glob : Whether to interpret glob patterns in files.
    single_col_name : If not None and the avro schema isn't a record, wrap
        values in a record with a single field called `single_col_name`.
    """
    lazy = scan_avro(
        sources,
        batch_size=batch_size,
        glob=glob,
        convert_logical_types=convert_logical_types,
        single_col_name=single_col_name,
    )
    if columns is not None:
        lazy = lazy.select(
            [pl.nth(c) if isinstance(c, int) else pl.col(c) for c in columns]
        )
    if row_index_name is not None:
        lazy = lazy.with_row_index(row_index_name, offset=row_index_offset)
    if n_rows is not None:
        lazy = lazy.limit(n_rows)
    res = lazy.collect()
    return res.rechunk() if rechunk else res
