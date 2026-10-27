"""Behavioural features per pseudo customer (uid), computed from strictly earlier transactions.

Rule (identical in batch and online): the history of a transaction at time t is every transaction
of the same uid with TransactionDT < t. Transactions in the same second do not see each other.
The 24 h window is [t - 24 h, t). Rows without a uid get NaN behavioural features.

Running sums are computed *per uid*, in time order (stable for ties), in both paths. This keeps the
numbers small (better float precision) and makes batch and online results match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_24H = 86_400
_KEY_STRIDE = 10**9  # > max TransactionDT + 24 h, so uid blocks never overlap in the composite key
# z-score only when log-amounts really vary (std > ~0.001, i.e. > ~0.1%); below that, rounding
# noise in the running sums (~1e-11) would create huge meaningless z-scores.
_MIN_VAR = 1e-6
Z_CAP = 10.0  # |z| > 10 just means "far outside the usual range"; capping keeps texts readable

BEHAVIOR_FEATURES = (
    "uid_n_prior",  # number of earlier transactions of this customer
    "uid_n_24h",  # earlier transactions in the last 24 h
    "uid_amt_sum_24h",  # total amount of those transactions
    "uid_amt_mean_prior",  # average amount of all earlier transactions
    "amt_to_uid_mean",  # this amount / that average
    "amt_z_uid",  # how unusual this amount is for the customer (log-scale z-score, capped at ±10)
    "secs_since_prev_uid",  # seconds since the customer's previous transaction
    "uid_new_device",  # 1 = device never seen before for this customer
    "uid_new_p_email",  # 1 = payer e-mail domain never seen before for this customer
)


def _empty(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=list(BEHAVIOR_FEATURES), dtype="float64")


def _derived(n_prior, s_amt, s_log, s_log2, amt, log_amt):
    """Mean / ratio / z-score from prior sums (vectorised or scalar, same formulas)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_amt = np.where(n_prior > 0, s_amt / n_prior, np.nan)
        ratio = np.where(mean_amt > 0, amt / mean_amt, np.nan)
        mean_log = np.where(n_prior > 0, s_log / n_prior, np.nan)
        var_log = np.where(n_prior >= 2, s_log2 / n_prior - mean_log**2, np.nan)
        z = np.where(var_log > _MIN_VAR, (log_amt - mean_log) / np.sqrt(var_log), np.nan)
        z = np.clip(z, -Z_CAP, Z_CAP)  # NaN stays NaN
    return mean_amt, ratio, z


def behavior_batch(
    t: pd.Series, amount: pd.Series, device: pd.Series, p_email: pd.Series, uid: pd.Series
) -> pd.DataFrame:
    """Vectorised behavioural features for a whole DataFrame (any row order)."""
    out = _empty(t.index)
    has = uid.notna().to_numpy()
    if not has.any():
        return out

    codes = pd.factorize(uid[has])[0].astype("int64")
    tt = t[has].to_numpy(dtype="int64")
    amt = amount[has].to_numpy(dtype="float64")
    dev = device[has].to_numpy(dtype=object)
    eml = p_email[has].to_numpy(dtype=object)

    order = np.lexsort((tt, codes))  # by uid, then time; stable for ties (keeps input order)
    c, ts, a = codes[order], tt[order], amt[order]
    la = np.log1p(a)
    keys = c * _KEY_STRIDE + ts

    # Per-uid inclusive running sums (reset at each uid start).
    is_start = np.r_[True, c[1:] != c[:-1]]
    grp = np.cumsum(is_start) - 1
    starts_of_grp = np.flatnonzero(is_start)

    def group_cumsum(x):
        # True per-uid sequential cumsum (same additions in the same order as the online path).
        # Do NOT use "global cumsum minus offset": that reintroduces large-number rounding.
        return pd.Series(x).groupby(grp, sort=False).cumsum().to_numpy()

    cs_amt, cs_log, cs_log2 = group_cumsum(a), group_cumsum(la), group_cumsum(la * la)

    start = starts_of_grp[grp]
    right = np.searchsorted(keys, c * _KEY_STRIDE + ts, side="left")  # first row with time >= t
    left24 = np.searchsorted(keys, c * _KEY_STRIDE + ts - WINDOW_24H, side="left")

    def prefix(cs, end):  # inclusive group sum of rows [start, end)
        return np.where(end > start, cs[np.maximum(end - 1, 0)], 0.0)

    n_prior = (right - start).astype("float64")
    n_24h = (right - left24).astype("float64")
    s_amt = prefix(cs_amt, right)
    sum_24h = s_amt - prefix(cs_amt, left24)
    mean_amt, ratio, z = _derived(
        n_prior, s_amt, prefix(cs_log, right), prefix(cs_log2, right), a, la
    )
    prev_t = np.where(right > start, ts[np.maximum(right - 1, 0)], -1)
    secs = np.where(right > start, ts - prev_t, np.nan).astype("float64")

    def first_seen_flag(values):
        v = values[order]
        present = pd.notna(v)
        flag = np.full(len(v), np.nan)
        if present.any():
            pair = pd.DataFrame({"c": c[present], "v": v[present], "t": ts[present]})
            first_t = pair.groupby(["c", "v"], sort=False)["t"].transform("min").to_numpy()
            flag[present] = (first_t == ts[present]).astype("float64")
        flag[n_prior == 0] = np.nan  # "new" is only meaningful when the customer has history
        return flag

    sorted_block = np.column_stack(
        [
            n_prior,
            n_24h,
            sum_24h,
            mean_amt,
            ratio,
            z,
            secs,
            first_seen_flag(dev),
            first_seen_flag(eml),
        ]
    )
    block = np.empty_like(sorted_block)
    block[order] = sorted_block
    out.loc[has, list(BEHAVIOR_FEATURES)] = block
    return out


def behavior_online(
    t: int, amount: float, device, p_email, uid, history: pd.DataFrame
) -> dict[str, float]:
    """Behavioural features for one transaction.

    `history` must hold this uid's transactions with TransactionDT < t, sorted by time (stable),
    with columns TransactionDT, TransactionAmt, device, p_email (same normalisation as batch).
    """
    nan = dict.fromkeys(BEHAVIOR_FEATURES, np.nan)
    if uid is None or (isinstance(uid, float) and np.isnan(uid)):
        return nan
    ts = history["TransactionDT"].to_numpy(dtype="int64")
    a = history["TransactionAmt"].to_numpy(dtype="float64")
    la = np.log1p(a)
    n = len(ts)
    # Same running-sum construction as the batch path.
    cs_amt, cs_log, cs_log2 = np.cumsum(a), np.cumsum(la), np.cumsum(la * la)
    left24 = int(np.searchsorted(ts, t - WINDOW_24H, side="left"))

    def prefix(cs, end):
        return float(cs[end - 1]) if end > 0 else 0.0

    s_amt = prefix(cs_amt, n)
    amt = float(amount)
    log_amt = float(np.log1p(amt))
    mean_amt, ratio, z = _derived(
        np.float64(n), s_amt, prefix(cs_log, n), prefix(cs_log2, n), amt, log_amt
    )

    def new_flag(value, col):
        if n == 0 or pd.isna(value):
            return np.nan
        seen = set(history[col].dropna().tolist())
        return float(value not in seen)

    return {
        "uid_n_prior": float(n),
        "uid_n_24h": float(n - left24),
        "uid_amt_sum_24h": s_amt - prefix(cs_amt, left24),
        "uid_amt_mean_prior": float(mean_amt),
        "amt_to_uid_mean": float(ratio),
        "amt_z_uid": float(z),
        "secs_since_prev_uid": float(t - ts[-1]) if n > 0 else np.nan,
        "uid_new_device": new_flag(device, "device"),
        "uid_new_p_email": new_flag(p_email, "p_email"),
    }
