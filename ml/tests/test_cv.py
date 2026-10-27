import numpy as np
import pytest

from fraudml.models.cv import forward_folds


def _days():
    # days 1..100 like the train period, 10 rows per day, shuffled row order
    rng = np.random.default_rng(0)
    return rng.permutation(np.repeat(np.arange(1, 101), 10))


def test_fold_windows_on_train_period():
    folds = forward_folds(_days())
    assert [f.val_days for f in folds] == [(56, 70), (71, 85), (86, 100)]
    assert [f.train_days for f in folds] == [(1, 48), (1, 63), (1, 78)]


def test_time_order_gap_and_no_overlap():
    day = _days()
    for f in forward_folds(day, gap_days=7):
        assert not set(f.train_idx) & set(f.val_idx)
        assert day[f.train_idx].max() + 7 < day[f.val_idx].min()
        assert day[f.val_idx].max() - day[f.val_idx].min() == 14


def test_expanding_windows():
    sizes = [len(f.train_idx) for f in forward_folds(_days())]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def test_last_fold_ends_on_last_day():
    day = _days()
    assert day[forward_folds(day)[-1].val_idx].max() == day.max()


def test_too_many_folds_raises():
    with pytest.raises(ValueError):
        forward_folds(_days(), n_folds=7, val_days=15)
