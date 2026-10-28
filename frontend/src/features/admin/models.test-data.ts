import type { Comparison } from './comparison'
import type { ModelVersion } from './modelsApi'

const side = (
  pr_auc: number,
  precision: number,
  recall: number,
  caught: number,
  savings: number,
  perDay: number,
) => ({
  n: 19941,
  n_fraud: 733,
  alerts: Math.round(perDay * 7),
  alerts_per_day: perDay,
  precision,
  recall,
  pr_auc,
  fraud_value_caught: caught,
  total_cost: 0,
  net_savings: savings,
})

/** The real comparison of the retrained model (reports/retrain_comparison.md). */
export const COMPARISON: Comparison = {
  window: ['2018-05-19', '2018-05-26'],
  champion_version: 'lgbm-2026.10.27-1',
  champion_threshold: 21.82,
  challenger_threshold: 17.92,
  champion: {
    overall: side(0.432, 0.257, 0.367, 0.645, 64265, 149.3),
    excl_fraud_customers: side(0.454, 0.247, 0.406, 0.674, 59809, 145.6),
  },
  challenger: {
    overall: side(0.532, 0.352, 0.457, 0.71, 72307, 136.0),
    excl_fraud_customers: side(0.519, 0.321, 0.462, 0.722, 65409, 127.9),
  },
  pr_auc_diff_bootstrap: { diff: 0.0998, low: 0.0641, high: 0.1403, share_positive: 1, days: 7 },
}

const base = {
  algorithm: 'lightgbm',
  anomaly_version: 'iforest-2026.10.27-1',
  policy: 'expected_loss',
  calibration_method: 'isotonic',
  trained_at: '2026-10-28T15:00:00',
  hyperparams: null,
  notes: null,
}

export const CHAMPION: ModelVersion = {
  ...base,
  id: 1,
  version: 'lgbm-2026.10.27-1',
  status: 'active',
  serving: true,
  threshold: 21.8209,
  created_at: '2026-10-28T08:00:00Z',
  training_data_window: {
    train: ['2017-12-02', '2018-03-12'],
    calibration: ['2018-03-19', '2018-04-21'],
  },
  metrics: { validation: { overall: { pr_auc: 0.35 } }, comparison: null },
}

export const CANDIDATE: ModelVersion = {
  ...base,
  id: 2,
  version: 'lgbm-2026.10.28-1',
  status: 'candidate',
  serving: false,
  threshold: 17.92,
  created_at: '2026-10-28T16:00:00Z',
  training_data_window: {
    train: ['2017-12-02', '2018-05-05'],
    calibration: ['2018-05-12', '2018-05-19'],
  },
  metrics: { validation: { overall: { pr_auc: 0.5 } }, comparison: COMPARISON },
}
