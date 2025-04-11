"""Test transitive property for writing and reading in a DataFrame."""

from collections.abc import Callable
from datetime import date, datetime
from io import BytesIO

import polars as pl
import pytest

from polars_fastavro import read_avro, write_avro

from .utils import frames_equal


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(pl.from_dict({}, schema={}), id="empty"),
        pytest.param(
            pl.from_dict({"col": [None, None]}, schema={"col": pl.Null}), id="nulls"
        ),
        pytest.param(
            pl.from_dict({"col": [True, False, None]}, schema={"col": pl.Boolean}),
            id="bools",
        ),
        pytest.param(
            pl.from_dict({"col": [-1, 0, 6, None]}, schema={"col": pl.Int32}),
            id="ints",
        ),
        pytest.param(
            pl.from_dict({"col": [-1, 0, 6, None]}, schema={"col": pl.Int64}),
            id="longs",
        ),
        pytest.param(
            pl.from_dict({"col": [date.today(), None]}, schema={"col": pl.Date}),
            id="dates",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [datetime.now(), None]},
                schema={"col": pl.Datetime("ms", "UTC")},
            ),
            id="datetime-ms",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [datetime.now(), None]},
                schema={"col": pl.Datetime("us", "UTC")},
            ),
            id="datetime-us",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [datetime.now(), None]},
                schema={"col": pl.Datetime("ms")},
            ),
            id="datetime-ms-local",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [datetime.now(), None]},
                schema={"col": pl.Datetime("us")},
            ),
            id="datetime-us-local",
        ),
        pytest.param(
            pl.from_dict({"col": [-1.0, 0.0, 6.0, None]}, schema={"col": pl.Float32}),
            id="floats",
        ),
        pytest.param(
            pl.from_dict({"col": [-1.0, 0.0, 6.0, None]}, schema={"col": pl.Float64}),
            id="doubles",
        ),
        pytest.param(
            pl.from_dict(
                {"col": ["a", "b", None]}, schema={"col": pl.Enum(["a", "b"])}
            ),
            id="enum",
        ),
        pytest.param(
            pl.from_dict({"col": [b"a", b"b", None]}, schema={"col": pl.Binary}),
            id="binary",
        ),
        pytest.param(
            pl.from_dict({"col": ["a", "b", None]}, schema={"col": pl.String}),
            id="string",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [["a", None], ["b", "c"], ["d"], None]},
                schema={"col": pl.List(pl.String)},
            ),
            id="list-string",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "col": [
                        {"s": "a", "i": 5},
                        {"s": "b", "i": None},
                        {"s": None, "i": 6},
                        {"s": None, "i": None},
                        None,
                    ]
                },
                schema={"col": pl.Struct({"s": pl.String, "i": pl.Int32})},
            ),
            id="struct",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "col": [
                        [[{"s": "a", "i": 5}, None], [], None],
                        [[{"s": "b", "i": None}]],
                        [[{"s": None, "i": 6}], [{"s": None, "i": None}]],
                        None,
                    ]
                },
                schema={
                    "col": pl.List(pl.List(pl.Struct({"s": pl.String, "i": pl.Int32})))
                },
            ),
            id="nested",
        ),
    ],
)
def test_transitive(frame: pl.DataFrame):
    """Test that frames can be serialized and deserialized."""
    buff = BytesIO()
    write_avro(frame, buff)
    buff.seek(0)
    dup = read_avro(buff)

    assert frames_equal(frame, dup)


@pytest.mark.parametrize(
    "write_func,read_func",
    [
        pytest.param(
            pl.DataFrame.write_avro, pl.read_avro, id="polars", marks=pytest.mark.xfail
        ),
        pytest.param(write_avro, read_avro, id="fastavro"),
    ],
)
def test_noncontiguous_chunks(
    write_func: Callable[[pl.DataFrame, BytesIO], None],
    read_func: Callable[[BytesIO], pl.DataFrame],
) -> None:
    """Test that non contiguous arrays can be written and read."""
    frame = pl.concat(
        [
            pl.from_dict({"split": [*range(3)]}),
            pl.from_dict({"split": [*range(3, 6)]}),
        ],
        rechunk=False,
    ).with_columns(contig=pl.int_range(pl.len()))
    buff = BytesIO()
    write_func(frame, buff)
    buff.seek(0)
    dup = read_func(buff)
    assert frames_equal(frame, dup)


@pytest.mark.parametrize(
    "write_func,read_func",
    [
        pytest.param(
            pl.DataFrame.write_avro, pl.read_avro, id="polars", marks=pytest.mark.xfail
        ),
        pytest.param(write_avro, read_avro, id="fastavro"),
    ],
)
def test_noncontiguous_arrays(
    write_func: Callable[[pl.DataFrame, BytesIO], None],
    read_func: Callable[[BytesIO], pl.DataFrame],
) -> None:
    """Test that non contiguous arrays can be written and read."""
    frame = pl.concat(
        [
            pl.from_dict({"split": [*range(3)]}),
            pl.from_dict({"split": [*range(3, 6)]}),
        ],
        rechunk=False,
    )
    buff = BytesIO()
    write_func(frame, buff)
    buff.seek(0)
    dup = read_func(buff)
    assert frames_equal(frame, dup)
