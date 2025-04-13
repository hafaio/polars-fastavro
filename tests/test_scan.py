"""Test scan functionality."""

from io import BytesIO

import fastavro
import polars as pl
import pytest

from polars_fastavro import read_avro, scan_avro, write_avro

from .utils import frames_equal


def test_scan_avro() -> None:
    """Test generic scan of files."""
    frame = scan_avro("resources/food.avro").with_row_index("row_index").collect()
    assert frame["row_index"].to_list() == [*range(27)]

    frame = (
        scan_avro("resources/food.avro")
        .with_row_index("row_index")
        .filter(pl.col("category") == pl.lit("vegetables"))  # type: ignore
        .collect()
    )
    assert frame["row_index"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    frame = (
        scan_avro("resources/food.avro")
        .with_row_index("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))  # type: ignore
        .collect()
    )
    assert frame["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_projection_pushdown_avro() -> None:
    """Test that projection is pushed down to scan."""
    file_path = "resources/food.avro"
    lazy = scan_avro(file_path).select(pl.col.calories)

    explain = lazy.explain()

    assert "simple π" not in explain
    assert "PROJECT 1/4 COLUMNS" in explain

    normal = lazy.collect()
    unoptimized = lazy.collect(no_optimization=True)
    assert frames_equal(normal, unoptimized)


def test_predicate_pushdown_avro() -> None:
    """Test that predicate is pushed down to scan."""
    file_path = "resources/food.avro"
    thresh = 80
    lazy = scan_avro(file_path).filter(pl.col("calories") > thresh)  # type: ignore

    explain = lazy.explain()

    assert "FILTER" not in explain
    assert """SELECTION: [(col("calories")) > (80)]""" in explain

    normal = lazy.collect()
    unoptimized = lazy.collect(no_optimization=True)
    assert frames_equal(normal, unoptimized)


def test_glob_n_rows() -> None:
    """Test that globbing and n_rows work."""
    file_path = "resources/*.avro"
    frame = scan_avro(file_path).limit(28).collect()

    # 27 rows from food.avro and 1 from grains.avro
    assert frame.shape == (28, 4)

    # take first and last rows
    assert frame[[0, 27]].to_dict(as_series=False) == {
        "category": ["vegetables", "rice"],
        "calories": [45, 9],
        "fats_g": [0.5, 0.0],
        "sugars_g": [2, 0.3],
    }


def test_many_files() -> None:
    """Test that scan works with many files."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [5, 12, 14]})
    write_avro(frame, buff)

    buffs = [BytesIO(buff.getvalue()) for _ in range(1023)]
    res = scan_avro(buffs).collect()
    reference = pl.from_dict({"x": [5, 12, 14] * 1023})
    assert frames_equal(res, reference)


def test_scan_nrows_empty() -> None:
    """Test that scan doesn't panic with n_rows set to 0."""
    file_path = "resources/food.avro"
    frame = scan_avro(file_path).head(0).collect()
    reference = read_avro(file_path).head(0)
    assert frames_equal(frame, reference)


def test_scan_filter_empty() -> None:
    """Test that scan doesn't panic when filter removes all rows."""
    file_path = "resources/food.avro"
    frame = scan_avro(file_path).filter(pl.col("category") == "empty").collect()  # type: ignore
    reference = read_avro(file_path).filter(pl.col("category") == "empty")  # type: ignore
    assert frames_equal(frame, reference)


def test_avro_list_arg() -> None:
    """Test that scan works when passing a list."""
    first = "resources/food.avro"
    second = "resources/grains.avro"

    frame = scan_avro([first, second]).collect()
    assert frame.shape == (30, 4)
    assert frame.row(-1) == ("corn", 99, 0.1, 10.4)
    assert frame.row(0) == ("vegetables", 45, 0.5, 2)


def test_glob_single_scan() -> None:
    """Test that globbing works with a single file."""
    file_path = "resources/food*.avro"
    frame = scan_avro(file_path)

    explain = frame.explain()

    assert explain.count("SCAN") == 1
    assert "UNION" not in explain


def test_scan_in_memory() -> None:
    """Test that scan works for in memory buffers."""
    frame = pl.from_dict({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    buff = BytesIO()
    write_avro(frame, buff)

    buff.seek(0)
    scanned = scan_avro(buff).collect()
    assert frames_equal(frame, scanned)

    buff.seek(0)
    scanned = scan_avro(buff).slice(1, 2).collect()
    assert frames_equal(frame.slice(1, 2), scanned)

    buff.seek(0)
    scanned = scan_avro(buff).slice(-1, 1).collect()
    assert frames_equal(frame.slice(-1, 1), scanned)

    other = BytesIO(buff.getvalue())

    buff.seek(0)
    scanned = scan_avro([buff, other]).collect()
    assert frames_equal(pl.concat([frame, frame]), scanned)

    buff.seek(0)
    other.seek(0)
    scanned = scan_avro([buff, other]).slice(1, 3).collect()
    assert frames_equal(pl.concat([frame, frame]).slice(1, 3), scanned)

    buff.seek(0)
    other.seek(0)
    scanned = scan_avro([buff, other]).slice(-4, 3).collect()
    assert frames_equal(pl.concat([frame, frame]).slice(-4, 3), scanned)


def test_large_scan(num: int = 200_000) -> None:
    """Test that scan works on large files over multiple chunks."""
    base_frame = pl.from_dict({"col": [*range(num)], "other": [*range(num)]})
    buff = BytesIO()
    write_avro(base_frame, buff)
    buff.seek(0)
    limit = 5
    result = (
        scan_avro(buff, batch_size=1024)
        .filter(pl.col("col") < limit)  # type:ignore
        .limit(limit + 1)
        .collect()
    )
    expected = pl.from_dict({"col": [*range(limit)], "other": [*range(limit)]})
    assert frames_equal(result, expected)


def test_read_options() -> None:
    """Test read works with options."""
    frame = read_avro(
        "resources/food.avro", row_index_name="row_index", columns=[1], n_rows=11
    )
    assert frame.shape == (11, 2)
    assert frame["row_index"].to_list() == [*range(11)]


def test_logical_types() -> None:
    """Test that we can read physical types."""
    buff = BytesIO()
    fastavro.writer(  # type: ignore
        buff,
        {
            "type": "record",
            "name": "schema",
            "fields": [
                {
                    "name": "decimal",
                    "type": {
                        "type": "bytes",
                        "logicalType": "decimal",
                        "precision": 1,
                        "scale": 1,
                    },
                },
                {
                    "name": "time-ms",
                    "type": {
                        "type": "int",
                        "logicalType": "time-millis",
                    },
                },
                {
                    "name": "time-us",
                    "type": {
                        "type": "long",
                        "logicalType": "time-micros",
                    },
                },
            ],
        },
        [{"decimal": b"decimal", "time-ms": 1, "time-us": 2}],
    )
    buff.seek(0)
    result = read_avro(buff, convert_logical_types=True)
    expected = pl.Schema(
        [("decimal", pl.Binary), ("time-ms", pl.Int32), ("time-us", pl.Int64)]
    )
    assert result.schema == expected

    buff.seek(0)
    with pytest.raises(Exception, match="without logical-type parsing"):
        read_avro(buff)


def test_timestamp_nanos() -> None:
    """Test that we can read timestamps with nanos."""
    buff = BytesIO()
    fastavro.writer(  # type: ignore
        buff,
        {
            "type": "record",
            "name": "schema",
            "fields": [
                {
                    "name": "local",
                    "type": {
                        "type": "long",
                        "logicalType": "local-timestamp-nanos",
                    },
                },
                {
                    "name": "absolute",
                    "type": {
                        "type": "long",
                        "logicalType": "timestamp-nanos",
                    },
                },
            ],
        },
        [{"local": 0, "absolute": 1}],
    )
    buff.seek(0)
    result = read_avro(buff, convert_logical_types=True)
    expected = pl.Schema(
        [("local", pl.Datetime("ns", None)), ("absolute", pl.Datetime("ns", "UTC"))]
    )
    assert result.schema == expected


def test_singleton_unions() -> None:
    """Test that we can read timestamps with nanos."""
    buff = BytesIO()
    fastavro.writer(  # type: ignore
        buff,
        {
            "type": "record",
            "name": "schema",
            "fields": [
                {
                    "name": "long",
                    "type": ["long"],
                },
                {
                    "name": "null",
                    "type": ["null"],
                },
            ],
        },
        [{"long": 0, "null": None}],
    )
    buff.seek(0)
    result = read_avro(buff, convert_logical_types=True)
    expected = pl.Schema([("long", pl.Int64), ("null", pl.Null)])
    assert result.schema == expected


def test_mixed_schema_err() -> None:
    """Test that mixing schemas raises."""
    one = pl.from_dict({"x": [2, 2]})
    buff_one = BytesIO()
    write_avro(one, buff_one)
    buff_one.seek(0)
    two = pl.from_dict({"y": ["a", "b"]})
    buff_two = BytesIO()
    write_avro(two, buff_two)
    buff_two.seek(0)
    lazy = scan_avro([buff_one, buff_two])
    with pytest.raises(
        Exception, match="schema of source 1 didn't match schema of source 0"
    ):
        lazy.collect()


def test_filename_in_err() -> None:
    """Test that invalid filename is reported in error."""
    lazy = scan_avro("does not exist", glob=False)
    with pytest.raises(Exception, match="does not exist"):
        lazy.collect()


def test_empty_sources() -> None:
    """Test that empty sources raises an error."""
    lazy = scan_avro([])
    with pytest.raises(Exception, match="sources were empty"):
        lazy.collect()


def test_invalid_dtype() -> None:
    """Test that error is thrown on non-record schema."""
    buff = BytesIO()
    fastavro.writer(  # type: ignore
        buff,
        {
            "type": "record",
            "name": "schema",
            "fields": [
                {"name": "field", "type": {"type": "fixed", "name": "fixed", "size": 0}}
            ],
        },
        [],
    )
    buff.seek(0)
    with pytest.raises(Exception, match="unhandled datatype:"):
        read_avro(buff)


def test_non_record_schema() -> None:
    """Test that error is thrown on non-record schema."""
    buff = BytesIO()
    fastavro.writer(buff, "int", [3, 7, 4])  # type: ignore
    buff.seek(0)
    with pytest.raises(Exception, match="top-level schema must be a record schema"):
        read_avro(buff)

    buff.seek(0)
    frame = read_avro(buff, single_col_name="col")
    expected = pl.from_dict({"col": [3, 7, 4]})
    assert frames_equal(frame, expected)
