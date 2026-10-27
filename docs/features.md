# Feature dictionary (curated feature set)

The model uses **59 curated features**, defined in `ml/fraudml/features/pipeline.py` (`FEATURES_CURATED`). Every feature is computed by the same code for training (batch) and for scoring a single transaction (online). A test checks that both give identical values.

**Rules that apply to every feature**
- **Past only:** behavioural features use only the same customer's transactions *strictly before* the current one. Transactions in the same second do not see each other.
- **No labels:** no feature uses `isFraud`, including past labels of the same customer.
- **No identifiers:** `TransactionID`, the pseudo customer key `uid` and raw timestamps are never features.
- **Train-only encoding:** category lists and frequencies are learned from the train period only. Values seen fewer than 100 times in train, or never seen, become `other`; missing values become `(missing)`.
- Missing numeric values stay missing (NaN). Gradient-boosted trees handle them natively, and missingness itself is informative in this dataset.

**Pseudo customer key** (used only to compute behavioural features): `uid = card1 + addr1 + (day − D1)`. `day − D1` estimates the day the card was first seen. Rows without `card1`, `addr1` or `D1` have no uid (about 11% of rows), and their behavioural features are missing.

## Time
| Feature | Meaning | Source |
|---|---|---|
| `hour` | Hour of day (0–23), relative to the dataset's reference clock (time zone unknown) | `TransactionDT` |
| `weekday` | Day of week (0 = Monday) | `TransactionDT` (reference 2017-12-01 = Friday) |

## Amount
| Feature | Meaning | Source |
|---|---|---|
| `amt_log` | log(1 + amount) | `TransactionAmt` |
| `amt_cents` | Cents part of the amount (0–99) | `TransactionAmt` |
| `amt_is_round` | 1 if the amount is a whole number of dollars | `TransactionAmt` |

## Product and card
| Feature | Meaning | Source |
|---|---|---|
| `ProductCD` | Product code (W, C, R, H, S) | `ProductCD` |
| `card4` | Card network (visa, mastercard, …) | `card4` |
| `card6` | Card type (debit, credit) | `card6` |
| `card1_freq` | Share of train transactions with the same card identifier (how common the card is) | `card1` |
| `card2_freq` | Same, for `card2` | `card2` |
| `addr1_freq` | Share of train transactions from the same billing region | `addr1` |

## E-mail
| Feature | Meaning | Source |
|---|---|---|
| `p_email` | Payer e-mail domain (rare domains → `other`) | `P_emaildomain` |
| `r_email` | Recipient e-mail domain (rare domains → `other`) | `R_emaildomain` |
| `email_same_domain` | 1 if payer and recipient domains are the same, 0 if different, missing if either is absent. In the data, *same* domain is riskier (9.7% vs 2.9% fraud). | both e-mail columns |
| `p_email_freq` | How common the payer domain is in train | `P_emaildomain` |

## Device
| Feature | Meaning | Source |
|---|---|---|
| `device_type` | desktop / mobile | `DeviceType` |
| `browser_family` | Browser family (chrome, safari, ie, edge, firefox, samsung, …). Versions are dropped because they drift quickly. | `id_31` |
| `os_family` | Operating system family (windows, ios, mac, android, linux) | `id_30` |
| `device_info_freq` | How common the device model / string is in train | `DeviceInfo` |

## Missing-data flags
| Feature | Meaning | Source |
|---|---|---|
| `has_identity` | 1 if identity / device data exists for the transaction (3.8× riskier in the data) | `id_01` |
| `has_uid` | 1 if a pseudo customer key could be built | `card1`, `addr1`, `D1` |
| `addr1_missing` | 1 if the billing region is missing (11.8% fraud vs 2.5%) | `addr1` |
| `dist1` | Distance field (meaning masked; often missing) | `dist1` |

## Customer behaviour (past only)
| Feature | Meaning (plain language) |
|---|---|
| `uid_n_prior` | Number of earlier transactions by this customer |
| `uid_n_24h` | Number of this customer's transactions in the previous 24 hours |
| `uid_amt_sum_24h` | Total amount this customer spent in the previous 24 hours |
| `uid_amt_mean_prior` | This customer's average amount over all earlier transactions |
| `amt_to_uid_mean` | This amount divided by the customer's usual (average) amount, e.g. 6 = "six times the usual amount" |
| `amt_z_uid` | How unusual the amount is for this customer (z-score on log amounts; needs at least 2 earlier transactions with varying amounts; capped at ±10) |
| `secs_since_prev_uid` | Seconds since this customer's previous transaction |
| `uid_new_device` | 1 = first time this device is used by this customer; 0 = seen before; missing if no history or no device data |
| `uid_new_p_email` | 1 = first time this payer e-mail domain is used by this customer; 0 = seen before |

## Masked features (meaning not published by the data provider)
In explanations these are grouped (e.g. "masked count features"), not described one by one.

| Features | Meaning | Source |
|---|---|---|
| `C1`–`C14` | Counts (e.g. how many addresses are linked to the payment card) | `C1`–`C14` |
| `D1n`, `D4n`, `D10n`, `D15n` | `day − D`: converts "days since an event" into "day the event happened", which stays stable over time (the raw D values drift) | `D1`, `D4`, `D10`, `D15` |
| `M1`–`M9` | Match flags (e.g. name on card vs address), values T / F (M4: M0–M2) | `M1`–`M9` |

## Extended set (comparison only)
`FEATURES_FULL` adds the raw `V1`–`V339` and `id_01`–`id_11` columns. It is used only for a single "curated vs full" comparison experiment, not by the scoring path.
