"""Training-serving skew test: online features == batch features for every synthetic row."""

from fraudml.features.encoder import FeatureEncoder
from fraudml.features.history import InMemoryHistory
from fraudml.features.pipeline import build_features_batch, raw_features_batch
from fraudml.features.skew import compare_rows
from tests.conftest import make_synthetic


def _setup(df):
    raw, _ = raw_features_batch(df)
    enc = FeatureEncoder(min_count=20).fit(raw)
    return enc, build_features_batch(df, enc), InMemoryHistory(df)


def test_online_equals_batch_for_every_row():
    df = make_synthetic(n=300, seed=0)  # includes same-second ties, missing uid, rare categories
    enc, batch, history = _setup(df)
    mismatches, max_abs = compare_rows(df, batch, history, enc)
    assert mismatches == [], mismatches[:5]
    assert max_abs <= 1e-9


def test_online_equals_batch_other_seeds():
    for seed in (1, 2):
        df = make_synthetic(n=150, seed=seed)
        enc, batch, history = _setup(df)
        mismatches, _ = compare_rows(df, batch, history, enc)
        assert mismatches == [], (seed, mismatches[:5])


def test_history_excludes_current_and_same_second(synthetic):
    _, _, history = _setup(synthetic)
    row = synthetic.iloc[9]  # rows 9, 10, 11 share the same second
    from fraudml.features.keys import make_uid

    uid = make_uid(synthetic.iloc[[9]]).iloc[0]
    if uid == uid:  # uid present
        hist = history.get(uid, before=int(row["TransactionDT"]))
        assert (hist["TransactionDT"] < row["TransactionDT"]).all()


def test_online_frame_keeps_categorical_dtypes():
    from fraudml.features.pipeline import (
        CATEGORICAL_FEATURES,
        compute_features_online,
        compute_features_online_frame,
    )

    df = make_synthetic(n=120, seed=3)
    enc, batch, history = _setup(df)
    txn = df.iloc[50].to_dict()
    frame = compute_features_online_frame(txn, history, enc)
    assert len(frame) == 1
    for col in CATEGORICAL_FEATURES:
        assert str(frame[col].dtype) == "category"
        assert list(frame[col].cat.categories) == list(batch[col].cat.categories)
    series = compute_features_online(txn, history, enc)
    assert [str(v) for v in frame.iloc[0]] == [str(v) for v in series]
