"""Plain-language helpers behind the analyst-facing explanation sentences."""

import math

import pytest

from fraudml.explain.templates import TEMPLATES, duration, money, rarity, share


@pytest.mark.parametrize(
    ("secs", "text"),
    [(45, "45 s"), (600, "10 min"), (3 * 3600, "3.0 h"), (5 * 86400, "5 days")],
)
def test_duration(secs, text):
    assert duration(secs) == text


@pytest.mark.parametrize(
    ("freq", "text"),
    [
        (0, "not seen in the training data"),
        (0.0001, "rare (0.01% of transactions)"),
        (0.002, "uncommon (0.20% of transactions)"),
        (0.25, "common (25.0% of transactions)"),
    ],
)
def test_rarity(freq, text):
    assert rarity(freq) == text


def test_money_and_share():
    assert money(1234.5) == "$1,234" and money(12.5) == "$12.50"
    assert share(0.004) == "0.40%" and share(0.5) == "50.0%"


# Derived from TransactionDT / TransactionAmt, which the scoring API requires: never missing.
FROM_REQUIRED_FIELDS = {"hour", "weekday", "amt_log", "amt_cents"}


def test_missing_values_get_a_readable_sentence():
    """Every other feature can be missing: its sentence must still read well (no "nan")."""
    for key, (label, fact) in TEMPLATES.items():
        if key in FROM_REQUIRED_FIELDS:
            continue
        for missing in (None, math.nan):
            text = fact(missing)
            assert isinstance(text, str) and text, key
            assert "nan" not in text.lower(), (key, text)
        assert label
