"""Plain-language wording for every feature and masked group.

`describe(key, value)` returns (label, fact): a short label and a sentence stating the fact
behind the feature ("Amount is 6.2x this customer's usual amount"). The SHAP direction and
strength are added by `sentence()`. Wording never claims causality: factors "raise" or "lower"
the model's risk score.

Masked (anonymised) columns are never named in the text; their families are described instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
STRENGTH_BINS = ((0.2, "slightly"), (0.7, "moderately"))  # |log-odds| below bound -> word
STRONG = "strongly"


def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or v == "(missing)"


def money(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def duration(secs: float) -> str:
    if secs < 120:
        return f"{secs:.0f} s"
    if secs < 7200:
        return f"{secs / 60:.0f} min"
    if secs < 2 * 86400:
        return f"{secs / 3600:.1f} h"
    return f"{secs / 86400:.0f} days"


def share(x: float) -> str:
    return f"{x * 100:.2f}%" if x < 0.01 else f"{x * 100:.1f}%"


def rarity(freq: float) -> str:
    if freq == 0:
        return "not seen in the training data"
    if freq < 0.0005:
        return f"rare ({share(freq)} of transactions)"
    if freq < 0.005:
        return f"uncommon ({share(freq)} of transactions)"
    return f"common ({share(freq)} of transactions)"


def _flag(yes: str, no: str, unknown: str) -> Callable:
    return lambda v: unknown if is_missing(v) else (yes if float(v) >= 0.5 else no)


def _category(label: str, what: str) -> Callable:
    def fact(v) -> str:
        if is_missing(v):
            return f"{what} unknown"
        if v == "other":
            return f"Uncommon {what.lower()}"
        return f"{what} is {v}"

    return fact


def _frequency(subject: str, missing_text: str) -> Callable:
    return lambda v: missing_text if is_missing(v) else f"{subject} is {rarity(float(v))}"


def _count(fmt: str, zero: str, missing: str = "No customer history available") -> Callable:
    def fact(v) -> str:
        if is_missing(v):
            return missing
        n = int(round(float(v)))
        return zero if n == 0 else fmt.format(n=n, s="" if n == 1 else "s")

    return fact


def _amount_ratio(v) -> str:
    if is_missing(v):
        return "No earlier amounts to compare with"
    return f"Amount is {float(v):.1f}x this customer's usual amount"


def _z(v) -> str:
    if is_missing(v):
        return "Not enough history to judge how unusual the amount is"
    z = float(v)
    side = "above" if z >= 0 else "below"
    return f"Amount is {abs(z):.1f} standard deviations {side} this customer's usual amounts"


# key -> (label, fact function)
TEMPLATES: dict[str, tuple[str, Callable]] = {
    "hour": ("Hour of day", lambda v: f"Made at {int(v):02d}:00 (dataset clock)"),
    "weekday": ("Day of week", lambda v: f"Made on a {WEEKDAYS[int(v)]}"),
    "amt_log": ("Amount", lambda v: f"Amount {money(math.expm1(float(v)))}"),
    "amt_cents": ("Cents", lambda v: f"Amount ends in .{int(v):02d}"),
    "amt_is_round": (
        "Round amount",
        _flag("Round dollar amount", "Amount is not a round number", "Amount roundness unknown"),
    ),
    "ProductCD": ("Product", _category("Product", "Product code")),
    "card4": ("Card network", _category("Card network", "Card network")),
    "card6": ("Card type", _category("Card type", "Card type")),
    "card1_freq": ("Card", _frequency("This card identifier", "Card identifier missing")),
    "card2_freq": ("Card group", _frequency("This card group", "Card group missing")),
    "addr1_freq": ("Billing region", _frequency("This billing region", "Billing region missing")),
    "p_email": ("Payer e-mail", _category("Payer e-mail", "Payer e-mail domain")),
    "r_email": ("Recipient e-mail", _category("Recipient e-mail", "Recipient e-mail domain")),
    "email_same_domain": (
        "E-mail match",
        _flag(
            "Payer and recipient use the same e-mail domain",
            "Payer and recipient e-mail domains differ",
            "E-mail domain match unknown (a domain is missing)",
        ),
    ),
    "p_email_freq": (
        "Payer e-mail popularity",
        _frequency("The payer e-mail domain", "No payer e-mail domain"),
    ),
    "device_type": ("Device type", _category("Device type", "Device type")),
    "browser_family": ("Browser", _category("Browser", "Browser")),
    "os_family": ("Operating system", _category("Operating system", "Operating system")),
    "device_info_freq": ("Device model", _frequency("This device model", "No device information")),
    "has_identity": (
        "Identity data",
        _flag(
            "Identity and device data present",
            "No identity or device data",
            "Identity data unknown",
        ),
    ),
    "has_uid": (
        "Customer key",
        _flag(
            "Customer recognised from card and billing address",
            "Customer could not be recognised (no history used)",
            "Customer key unknown",
        ),
    ),
    "addr1_missing": (
        "Billing region",
        _flag("Billing region is missing", "Billing region is present", "Billing region unknown"),
    ),
    "uid_n_prior": (
        "Customer history",
        _count(
            "{n} earlier transaction{s} from this customer",
            "First transaction seen for this customer",
        ),
    ),
    "uid_n_24h": (
        "Last 24 h activity",
        _count(
            "{n} transaction{s} from this customer in the previous 24 h",
            "No other transaction from this customer in the previous 24 h",
        ),
    ),
    "uid_amt_sum_24h": (
        "Last 24 h spending",
        lambda v: (
            "No customer history available"
            if is_missing(v)
            else f"{money(float(v))} spent by this customer in the previous 24 h"
        ),
    ),
    "uid_amt_mean_prior": (
        "Usual amount",
        lambda v: (
            "No earlier amounts for this customer"
            if is_missing(v)
            else f"This customer's usual amount is {money(float(v))}"
        ),
    ),
    "amt_to_uid_mean": ("Amount vs usual", _amount_ratio),
    "amt_z_uid": ("Amount unusualness", _z),
    "secs_since_prev_uid": (
        "Time since previous",
        lambda v: (
            "No previous transaction from this customer"
            if is_missing(v)
            else f"{duration(float(v))} since this customer's previous transaction"
        ),
    ),
    "uid_new_device": (
        "New device",
        _flag(
            "First transaction from this device for this customer",
            "Device already used by this customer",
            "Device history unknown",
        ),
    ),
    "uid_new_p_email": (
        "New e-mail domain",
        _flag(
            "First time this customer uses this e-mail domain",
            "E-mail domain already used by this customer",
            "E-mail history unknown",
        ),
    ),
    # Masked families (grouped contributions; value is not shown)
    "masked_counts": (
        "Counting signals (masked)",
        lambda v: "Provider counting signals (anonymised)",
    ),
    "masked_timing": (
        "Account timing (masked)",
        lambda v: "Provider account timing signals (anonymised)",
    ),
    "masked_match": (
        "Match checks (masked)",
        lambda v: "Provider identity match checks (anonymised)",
    ),
    "masked_distance": ("Distance (masked)", lambda v: "Provider distance signal (anonymised)"),
}


def strength(contribution: float) -> str:
    size = abs(contribution)
    for bound, word in STRENGTH_BINS:
        if size < bound:
            return word
    return STRONG


def direction(contribution: float) -> str:
    return "raises risk" if contribution > 0 else "lowers risk"


def describe(key: str, value) -> tuple[str, str]:
    label, fact = TEMPLATES[key]
    return label, fact(value)


def sentence(key: str, value, contribution: float) -> str:
    _, fact = describe(key, value)
    return f"{fact}: {strength(contribution)} {direction(contribution)}"
