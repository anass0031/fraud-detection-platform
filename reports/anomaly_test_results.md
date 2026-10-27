# Anomaly safety net: test-period check (iforest-2026.10.27-1 with lgbm-2026.10.27-1, run once on 2026-10-27)

|                 |   extra_alerts_per_day |   extra_alerts_max_day |   extra_frauds |   extra_precision |   extra_lift |   share_of_missed_frauds_caught |   main_fraud_value_caught |   combined_fraud_value_caught |   delta_net_savings |
|:----------------|-----------------------:|-----------------------:|---------------:|------------------:|-------------:|--------------------------------:|--------------------------:|------------------------------:|--------------------:|
| overall         |                 2.8286 |                 8.0000 |         1.0000 |            0.0101 |       0.2941 |                          0.0005 |                    0.6546 |                        0.6549 |           -516.0000 |
| excl_fraud_uids |                 2.8000 |                 8.0000 |         1.0000 |            0.0102 |       0.3282 |                          0.0005 |                    0.6690 |                        0.6694 |           -509.0000 |

Isolation Forest alone: PR-AUC 0.0742; 0.15% of rows above the cut-off, precision 0.1575 (fraud rate 0.0343).
