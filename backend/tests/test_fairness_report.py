import pandas as pd
import pytest

from app.scripts.fairness_report import MIN_ROWS, group_table, to_markdown


def test_group_rates_and_small_groups_merged():
    big = MIN_ROWS
    df = pd.DataFrame(
        {
            "product_code": ["W"] * big + ["C"] * big + ["Z"] * 3,
            # W: 10 frauds, 20 alerts (5 true) -> precision 0.25, recall 0.5
            "fraud": [True] * 10 + [False] * (big - 10) + [True] * big + [False] * 3,
            "alert": [True] * 5
            + [False] * 5
            + [True] * 15
            + [False] * (big - 25)
            + [True] * big
            + [False] * 3,
        }
    )
    t = group_table(df, "product_code").set_index("group")
    assert set(t.index) == {"W", "C", "other (small groups)"}
    w = t.loc["W"]
    assert w["transactions"] == big
    assert w["precision"] == pytest.approx(5 / 20)
    assert w["recall"] == pytest.approx(5 / 10)
    assert w["false-alarm rate"] == pytest.approx(15 / (big - 10))
    assert t.loc["C", "false-alarm rate"] != t.loc["C", "false-alarm rate"]  # no legit rows: NaN
    md = to_markdown(t.reset_index())
    assert "| W | 500 |" in md and "—" in md
