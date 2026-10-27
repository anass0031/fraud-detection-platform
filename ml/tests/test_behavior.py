import numpy as np
import pandas as pd
import pytest

from fraudml.features.behavior import BEHAVIOR_FEATURES, behavior_batch, behavior_online

H = 3600


def _toy():
    # uid A: 4 transactions (two in the same second), uid B: 2, one row without uid.
    return pd.DataFrame(
        {
            "t": [0, 10 * H, 10 * H, 30 * H, 5 * H, 6 * H, 7 * H],
            "amt": [100.0, 50.0, 70.0, 200.0, 10.0, 10.0, 999.0],
            "device": ["d1", "d1", "d2", "d2", None, "x", "z"],
            "email": ["g", "g", "g", "y", "h", "h", None],
            "uid": ["A", "A", "A", "A", "B", "B", np.nan],
        },
        index=[10, 11, 12, 13, 14, 15, 16],
    )


def _batch(df):
    return behavior_batch(df["t"], df["amt"], df["device"], df["email"], df["uid"])


def test_counts_sums_and_ties():
    out = _batch(_toy())
    # First A transaction has no history.
    assert out.loc[10, "uid_n_prior"] == 0 and np.isnan(out.loc[10, "uid_amt_mean_prior"])
    # The two A transactions at 10h only see the one at 0h (ties do not see each other).
    for i in (11, 12):
        assert out.loc[i, "uid_n_prior"] == 1
        assert out.loc[i, "uid_n_24h"] == 1
        assert out.loc[i, "uid_amt_sum_24h"] == 100.0
        assert out.loc[i, "secs_since_prev_uid"] == 10 * H
    assert out.loc[11, "amt_to_uid_mean"] == pytest.approx(0.5)
    # At 30h: 3 earlier transactions; the 24h window [6h, 30h) contains the two at 10h.
    assert out.loc[13, "uid_n_prior"] == 3
    assert out.loc[13, "uid_n_24h"] == 2
    assert out.loc[13, "uid_amt_sum_24h"] == 120.0
    assert out.loc[13, "uid_amt_mean_prior"] == pytest.approx(220 / 3)


def test_new_device_and_email_flags():
    out = _batch(_toy())
    assert np.isnan(out.loc[10, "uid_new_device"])  # no history -> not applicable
    assert out.loc[11, "uid_new_device"] == 0  # d1 seen at 0h
    assert out.loc[12, "uid_new_device"] == 1  # d2 only seen in the same second -> still new
    assert out.loc[13, "uid_new_device"] == 0
    assert out.loc[13, "uid_new_p_email"] == 1  # "y" never seen before
    assert np.isnan(out.loc[14, "uid_new_device"])  # device missing


def test_no_uid_gives_nan():
    out = _batch(_toy())
    assert out.loc[16].isna().all()


def test_zscore_needs_two_prior_and_variance():
    out = _batch(_toy())
    assert np.isnan(out.loc[11, "amt_z_uid"])  # only one prior transaction
    assert np.isnan(out.loc[15, "amt_z_uid"])  # B: one prior
    assert np.isfinite(out.loc[13, "amt_z_uid"])


def test_row_order_does_not_matter():
    df = _toy()
    shuffled = df.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(_batch(df), _batch(shuffled).loc[df.index])


def test_future_rows_do_not_change_past_features():
    df = _toy()
    full = _batch(df)
    cut = 10 * H
    past = df[df["t"] <= cut]
    pd.testing.assert_frame_equal(full.loc[past.index], _batch(past))


def test_online_matches_batch_on_toy():
    df = _toy()
    batch = _batch(df)
    hist_all = df.rename(
        columns={"t": "TransactionDT", "amt": "TransactionAmt", "email": "p_email"}
    )
    for i, row in df.iterrows():
        h = hist_all[(hist_all["uid"] == row["uid"]) & (hist_all["TransactionDT"] < row["t"])]
        h = h.sort_values("TransactionDT", kind="stable")
        online = behavior_online(row["t"], row["amt"], row["device"], row["email"], row["uid"], h)
        np.testing.assert_allclose(
            [online[f] for f in BEHAVIOR_FEATURES], batch.loc[i].to_numpy(), rtol=1e-12
        )


def test_zscore_is_capped():
    # Many identical amounts then a slightly different one -> tiny variance -> huge raw z.
    n = 50
    df = pd.DataFrame(
        {
            "t": list(range(0, n * 10, 10)) + [n * 10 + 10, n * 10 + 20],
            "amt": [49.0] * n + [49.5, 60.0],
            "device": [None] * (n + 2),
            "email": [None] * (n + 2),
            "uid": ["A"] * (n + 2),
        }
    )
    z = _batch(df)["amt_z_uid"]
    assert z.iloc[-1] == 10.0


def test_running_sums_do_not_depend_on_other_customers():
    """Large amounts of OTHER customers must not change this customer's features at all
    (guards against computing per-uid sums as global cumsum minus offset)."""
    df = _toy()
    big = pd.DataFrame(
        {
            "t": np.arange(100) * 60,
            "amt": np.full(100, 9.9e6),
            "device": [None] * 100,
            "email": [None] * 100,
            "uid": ["BIG"] * 100,
        },
        index=range(1000, 1100),
    )
    alone = _batch(df)
    mixed = _batch(pd.concat([big, df]))  # BIG customer sorts before A and B
    pd.testing.assert_frame_equal(alone, mixed.loc[df.index], check_exact=True)
