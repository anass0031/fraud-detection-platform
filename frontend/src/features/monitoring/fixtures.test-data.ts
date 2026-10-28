import type { Alerts, Drift, Feedback, Performance, PerformanceTotals } from './api'

export function totals(over: Partial<PerformanceTotals> = {}): PerformanceTotals {
  return {
    first_day: '2018-04-28',
    last_day: '2018-04-30',
    days: 3,
    n: 7558,
    n_fraud: 210,
    alerts: 382,
    alerts_per_day: 127.33,
    precision: 0.215,
    recall: 0.371,
    pr_auc: 0.356,
    fraud_value_caught: 0.583,
    total_cost: 20_000,
    no_model_cost: 34_315,
    net_savings: 14_315,
    ...over,
  }
}

export const PERFORMANCE: Performance = {
  model_version: 'lgbm-2026.10.27-1',
  days: ['2018-04-28', '2018-04-29', '2018-04-30'].map((day, i) => ({
    day,
    n: 2500,
    n_fraud: 70,
    alerts: 127,
    tp: 27,
    fp: 100,
    fn: 43,
    precision: 0.2 + i * 0.01,
    recall: 0.37,
    pr_auc: i === 1 ? null : 0.35,
    precision_at_capacity: 0.2,
    fraud_value_caught: 0.58,
    total_cost: 6_000,
    no_model_cost: 11_000,
    net_savings: 5_000,
  })),
  total: totals(),
  last_7_days: totals({ precision: 0.25, alerts_per_day: 140 }),
}

export const DRIFT: Drift = {
  model_version: 'lgbm-2026.10.27-1',
  days: ['2018-04-28', '2018-04-29'],
  features: [
    {
      feature: 'D1n',
      max_psi: 1.94,
      mean_psi: 1.9,
      latest_psi: 1.94,
      latest_severity: 'significant',
      days_significant: 2,
      cells: [
        { day: '2018-04-28', psi: 1.86, severity: 'significant' },
        { day: '2018-04-29', psi: 1.94, severity: 'significant' },
      ],
    },
    {
      feature: 'M3',
      max_psi: 0.14,
      mean_psi: 0.1,
      latest_psi: 0.14,
      latest_severity: 'moderate',
      days_significant: 0,
      cells: [{ day: '2018-04-29', psi: 0.14, severity: 'moderate' }],
    },
  ],
  score: [
    { day: '2018-04-28', psi: 0.014, ks_stat: 0.05, severity: 'stable' },
    { day: '2018-04-29', psi: 0.007, ks_stat: 0.04, severity: 'stable' },
  ],
  thresholds: { moderate: 0.1, significant: 0.25 },
}

export const ALERTS: Alerts = {
  days: [
    { day: '2018-04-28', model_alerts: 154, unusual: 2, open: 153, closed: 3 },
    { day: '2018-04-29', model_alerts: 101, unusual: 0, open: 101, closed: 0 },
  ],
  backlog: { new: 438, assigned: 2, escalated: 1 },
  capacity_per_day: 150,
}

export const FEEDBACK: Feedback = {
  decisions: { confirmed_fraud: 2, false_positive: 1, escalate: 1 },
  cases_closed: 3,
  confirmed_share: 2 / 3,
  checked_against_labels: 3,
  matching_labels: 1,
  mean_hours_to_decision: 0.76,
}
