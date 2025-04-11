"""Test write functionality."""

from datetime import datetime
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest

from polars_fastavro import read_avro, write_avro

from .utils import frames_equal


def test_binary_write() -> None:
    """Test writing to a buffer."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [1]})
    write_avro(frame, buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_chunked_binary_write() -> None:
    """Test writing to a buffer."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [1, 2]})
    write_avro(frame, buff, batch_size=1)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_empty_write() -> None:
    """Test writing an empty frame."""
    buff = BytesIO()
    frame = pl.from_dict({"x": []}, schema={"x": pl.Int32})
    write_avro(frame, buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_struct_write() -> None:
    """Test writing a struct."""
    buff = BytesIO()
    frame = pl.from_dict(
        {"x": [[1, "a"]]},
        schema={
            "x": pl.Struct(
                {
                    "a": pl.Int32,
                    "b": pl.String,
                }
            )
        },
    )
    write_avro(frame, buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_file_write(tmp_path: Path) -> None:
    """Test writing to a file."""
    path = tmp_path / "test.avro"

    frame = pl.from_dict({"x": [1]})
    write_avro(frame, path)
    duplicate = read_avro(path)
    assert frames_equal(frame, duplicate)


def test_int_promotion() -> None:
    """Test you can write ints with promotion."""
    buff = BytesIO()
    frame = pl.from_dict(
        {
            "i8": [1, -2],
            "i16": [-3, None],
            "u8": [None, 4],
            "u16": [7, None],
            "u32": [100, 0],
        },
        schema={
            "i8": pl.Int8,
            "i16": pl.Int16,
            "u8": pl.UInt8,
            "u16": pl.UInt16,
            "u32": pl.UInt32,
        },
    )
    write_avro(frame, buff)
    buff.seek(0)
    promoted = frame.select(
        pl.col("i8").cast(pl.Int32),
        pl.col("i16").cast(pl.Int32),
        pl.col("u8").cast(pl.Int32),
        pl.col("u16").cast(pl.Int32),
        pl.col("u32").cast(pl.Int64),
    )
    dup = read_avro(buff)
    assert frames_equal(dup, promoted)


def test_no_int_promotion() -> None:
    """Test exception when writing ints without promotion."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [1]}, schema={"x": pl.Int8})
    with pytest.raises(Exception, match="unsupported dtype: Int8"):
        write_avro(frame, buff, promote_ints=False)


def test_array_promotion() -> None:
    """Test arrays can be promoted."""
    buff = BytesIO()
    frame = pl.from_dict(
        {"x": [[1, 2], [None, 4], None]}, schema={"x": pl.Array(pl.Int32, 2)}
    )
    write_avro(frame, buff)
    buff.seek(0)
    dup = read_avro(buff)
    assert frames_equal(dup, frame.select(x=pl.col("x").arr.to_list()))


def test_no_array_promotion() -> None:
    """Test exception when writing arrays without promotion."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [[1]]}, schema={"x": pl.Array(pl.Int32, 1)})
    with pytest.raises(Exception, match="unsupported dtype: Array"):
        write_avro(frame, buff, promote_array=False)


def test_invalid_datetime() -> None:
    """Test that exception is raised for invalid Datetime."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [datetime.now()]}, schema={"x": pl.Datetime("ns")})
    with pytest.raises(Exception, match="unsupported dtype: Datetime"):
        write_avro(frame, buff)

    frame = pl.from_dict(
        {"x": [datetime.now()]}, schema={"x": pl.Datetime("us", "GMT")}
    )
    with pytest.raises(Exception, match="unsupported dtype: Datetime"):
        write_avro(frame, buff)
