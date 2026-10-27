import pandas as pd
import pytest

from fraudml import config
from fraudml.data.split import assign_period, period_bounds_seconds


def _dt(ts: str) -> int:
    return int((pd.Timestamp(ts) - pd.Timestamp(config.REFERENCE_DATETIME)).total_seconds())


def test_bounds_are_contiguous_and_ordered():
    bounds = list(period_bounds_seconds().values())
    for (_, end), (start, _) in zip(bounds, bounds[1:], strict=False):
        assert end == start
    assert all(a < b for a, b in bounds)


@pytest.mark.parametrize(
    "ts, expected",
    [
        ("2017-12-02 00:00:00", "train"),
        ("2018-03-11 23:59:59", "train"),
        ("2018-03-12 00:00:00", "gap_1"),
        ("2018-03-18 23:59:59", "gap_1"),
        ("2018-03-19 00:00:00", "validation"),
        ("2018-04-20 23:59:59", "validation"),
        ("2018-04-21 00:00:00", "gap_2"),
        ("2018-04-28 00:00:00", "test"),
        ("2018-06-01 23:59:59", "test"),
    ],
)
def test_boundaries(ts, expected):
    assert assign_period(pd.Series([_dt(ts)])).iloc[0] == expected


def test_out_of_range_raises():
    with pytest.raises(ValueError, match="outside"):
        assign_period(pd.Series([_dt("2018-06-02 00:00:00")]))
