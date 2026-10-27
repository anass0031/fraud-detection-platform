# Test-period results (lgbm-2026.10.27-1, evaluated once on 2026-10-27)

Alert policy `expected_loss` with threshold 21.82, fixed on validation.
`pr_auc` in the first table is computed on the alert score (expected loss).

| population      |   pr_auc |   alerts_per_day |   max_alerts_per_day |   precision_at_threshold |   recall_at_threshold |   fraud_value_caught_at_threshold |   total_cost |   no_model_cost |   net_savings |
|:----------------|---------:|-----------------:|---------------------:|-------------------------:|----------------------:|----------------------------------:|-------------:|----------------:|--------------:|
| overall         |   0.2282 |         152.9714 |                  210 |                   0.2503 |                0.3902 |                            0.6546 |  213153.6880 |     516308.6920 |   303155.0040 |
| excl_fraud_uids |   0.2188 |         149.4286 |                  202 |                   0.2358 |                0.3993 |                            0.6690 |  189377.3380 |     468982.8420 |   279605.5040 |

## Calibrated probability (production model)

|                 |   pr_auc |   roc_auc |   precision_at_capacity |   fraud_value_caught_at_capacity |   brier |   pr_auc_uncalibrated |   mean_prob |   fraud_rate |
|:----------------|---------:|----------:|------------------------:|---------------------------------:|--------:|----------------------:|------------:|-------------:|
| overall         |   0.4543 |    0.8975 |                  0.3617 |                           0.4741 |  0.0241 |                0.4674 |      0.0423 |       0.0343 |
| excl_fraud_uids |   0.4552 |    0.8974 |                  0.3358 |                           0.4854 |  0.0220 |              nan      |    nan      |     nan      |

## Threshold-free comparison (all models trained on the train period)

| model                                   | population      |   pr_auc |   roc_auc |   precision_at_capacity |   fraud_value_caught_at_capacity |
|:----------------------------------------|:----------------|---------:|----------:|------------------------:|---------------------------------:|
| production LightGBM (lgbm-2026.10.27-1) | overall         |   0.4543 |    0.8975 |                  0.3617 |                           0.4741 |
| production LightGBM (lgbm-2026.10.27-1) | excl_fraud_uids |   0.4552 |    0.8974 |                  0.3358 |                           0.4854 |
| XGBoost defaults                        | overall         |   0.4403 |    0.8754 |                  0.3368 |                           0.4435 |
| XGBoost defaults                        | excl_fraud_uids |   0.4343 |    0.8759 |                  0.3091 |                           0.4439 |
| logistic regression                     | overall         |   0.3720 |    0.8517 |                  0.3048 |                           0.3721 |
| logistic regression                     | excl_fraud_uids |   0.3963 |    0.8659 |                  0.3013 |                           0.4027 |
| rule: amount > train p99                | overall         |   0.0363 |    0.4739 |                  0.0516 |                           0.4312 |
| rule: amount > train p99                | excl_fraud_uids |   0.0329 |    0.4690 |                  0.0490 |                           0.4499 |

## Weekly stability

|   week | first_day   |     n |   fraud_rate |   pr_auc_prob |   alerts_per_day |   precision_at_threshold |   fraud_value_caught_at_threshold |   net_savings |   no_model_cost |
|-------:|:------------|------:|-------------:|--------------:|-----------------:|-------------------------:|----------------------------------:|--------------:|----------------:|
|      1 | 2018-04-28  | 21443 |       0.0308 |        0.5037 |         150.4286 |                   0.2374 |                            0.6183 |    47445.0890 |      87845.2500 |
|      2 | 2018-05-05  | 19811 |       0.0285 |        0.4096 |         142.7143 |                   0.2352 |                            0.6384 |    49158.0430 |      87216.7260 |
|      3 | 2018-05-12  | 21038 |       0.0347 |        0.4264 |         162.8571 |                   0.2316 |                            0.6497 |    58280.4830 |     101177.9900 |
|      4 | 2018-05-19  | 19941 |       0.0368 |        0.4320 |         149.2857 |                   0.2574 |                            0.6446 |    64264.5910 |     110216.7740 |
|      5 | 2018-05-26  | 17752 |       0.0421 |        0.5116 |         159.5714 |                   0.2883 |                            0.7022 |    84006.7980 |     129851.9520 |
