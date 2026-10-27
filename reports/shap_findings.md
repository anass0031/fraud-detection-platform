# SHAP explanations: findings

Explanations for the production model `lgbm-2026.10.27-1`, computed with LightGBM's built-in exact TreeSHAP (`pred_contrib=True`). Code: `ml/fraudml/explain/`. Examples: [`shap_examples.md`](shap_examples.md). All numbers come from the validation period.

## What an explanation contains

For each transaction: the base value, the raw log-odds, the calibrated fraud probability, and the **top 10 factors** ordered by size of contribution. Each factor has a key, a readable label, its value, its contribution, direction (raises / lowers risk), strength (slightly < 0.2 ≤ moderately < 0.7 ≤ strongly, in log-odds), a `masked` flag and a sentence. For example:

> Amount $554: moderately raises risk
> 24.5 h since this customer's previous transaction: moderately raises risk
> Card type is credit: strongly raises risk

Masked (anonymised) columns are never named. They are summed into four families: counting signals (`C1–C14`), account timing (`D1n, D4n, D10n, D15n`), identity match checks (`M1–M9`) and distance (`dist1`). The object is JSON-ready and is what the scoring API will store in `predictions.shap_values`.

## Checks

| Check | Result |
|---|---|
| Base value + contributions = raw log-odds (1,000 rows) | max error 2e-14 |
| Same values as `shap.TreeExplainer` | max difference 0 |
| Time to explain one transaction | 18 ms (target < 50 ms) |
| Every curated feature has wording; no masked column name in any sentence | enforced by tests |

## Findings

1. **The anonymised families drive the score most.** By mean |SHAP| they rank account timing 0.54, counting signals 0.42, match checks 0.27. Then come readable features: amount 0.22, card type 0.18, time since the previous transaction 0.16, payer e-mail 0.15, card frequency 0.14 ([figure](figures/shap_importance.png)). So explanations often lead with an anonymised family. Each factor carries a `masked` flag, so the case page can list readable reasons first. The stored top 10 contain enough readable factors for that.
2. **The directions are plausible** ([beeswarm](figures/shap_beeswarm.png)). Larger amounts raise risk. A short time since the customer's previous transaction raises risk. Credit cards raise risk. Very common cards lower risk, while rare or unseen ones raise it.
3. **The 10 example cases read sensibly.** Caught frauds show credit card, uncommon card, missing payer e-mail and larger amounts. False alarms look suspicious for understandable reasons (large amount, credit card, unknown payer e-mail), which is what an analyst needs to judge them. The two missed frauds differ:
   - One scores only 0.3%: nothing in its features looks risky.
   - One scores **90% but is only $15**, so its expected loss ($13.52) stays under the $21.82 alert threshold. This follows from the cost-based rule. Whether very high-probability small cases should still be queued is an open question for the review queue.
4. **Frequency features carry identity hints.** "This card identifier is uncommon" partly re-identifies cards. This was already seen during model selection, and the explanations make it visible.
5. **What SHAP explains:** contributions explain the model's risk score (log-odds), not causes, and not the amount-based alert rule. The case page will show the amount and expected loss as plain facts next to the factors, and will say "contributed to the score", never "caused".
