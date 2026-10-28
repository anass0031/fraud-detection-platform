/** Pure helpers that turn monitoring API answers into what the dashboard draws. */
import { money } from '../../utils/format'
import type { Alerts, Drift, Performance, PerformanceTotals } from './api'

export type Kpi = {
  key: string
  label: string
  value: string
  previous: string | null // the whole period, for comparison
  delta: number | null // last 7 days minus whole period (same unit as the raw value)
  better: 'up' | 'down'
  help: string
}

const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const num = (v: number | null | undefined, digits = 0) => (v == null ? '—' : v.toFixed(digits))

/** KPI cards: the last 7 replayed days, compared with the whole replayed period. */
export function kpis(last: PerformanceTotals, all: PerformanceTotals): Kpi[] {
  const card = (
    key: keyof PerformanceTotals,
    label: string,
    fmt: (v: number | null) => string,
    better: 'up' | 'down',
    help: string,
  ): Kpi => {
    const a = last[key] as number | null
    const b = all[key] as number | null
    return {
      key,
      label,
      value: fmt(a),
      previous: fmt(b),
      delta: a != null && b != null ? a - b : null,
      better,
      help,
    }
  }
  return [
    card('precision', 'Precision of alerts', pct, 'up', 'Share of alerts that were fraud'),
    card('recall', 'Recall', pct, 'up', 'Share of frauds that were alerted'),
    card(
      'pr_auc',
      'PR-AUC',
      (v) => num(v, 3),
      'up',
      'Ranking quality, independent of the threshold',
    ),
    card(
      'fraud_value_caught',
      'Fraud $ caught',
      pct,
      'up',
      'Share of fraud dollars inside the alerts',
    ),
    card(
      'net_savings',
      'Net savings',
      (v) => (v == null ? '—' : money(v)),
      'up',
      'Fraud losses avoided minus review and friction costs (sum over the days)',
    ),
    card(
      'alerts_per_day',
      'Alerts per day',
      (v) => num(v, 0),
      'down',
      'Model alerts per day; analysts can review the daily capacity',
    ),
  ]
}

/** PSI band of a value, using the API's thresholds. */
export function psiBand(
  psi: number,
  t: Drift['thresholds'],
): 'stable' | 'moderate' | 'significant' {
  if (psi < t.moderate) return 'stable'
  if (psi <= t.significant) return 'moderate'
  return 'significant'
}

export const BAND_COLOR = {
  stable: 'var(--mantine-color-teal-1)',
  moderate: 'var(--mantine-color-yellow-3)',
  significant: 'var(--mantine-color-red-5)',
} as const

export type HeatmapRow = {
  name: string
  label: string
  maxPsi: number
  cells: Map<string, number> // day -> psi
}

const FEATURE_LABEL: Record<string, string> = {
  __score__: 'Model score',
  amt_log: 'Amount (log)',
  D1n: 'Days since D1 (normalised)',
  D4n: 'Days since D4 (normalised)',
  D10n: 'Days since D10 (normalised)',
  D15n: 'Days since D15 (normalised)',
  uid_n_prior: 'Earlier transactions of the customer',
}

/** Model score first, then the `top` features with the highest PSI. */
export function heatmapRows(drift: Drift, top = 15): HeatmapRow[] {
  const score: HeatmapRow = {
    name: '__score__',
    label: FEATURE_LABEL.__score__,
    maxPsi: Math.max(0, ...drift.score.map((s) => s.psi)),
    cells: new Map(drift.score.map((s) => [s.day, s.psi])),
  }
  const features = drift.features.slice(0, top).map((f) => ({
    name: f.feature,
    label: FEATURE_LABEL[f.feature] ?? f.feature,
    maxPsi: f.max_psi,
    cells: new Map(f.cells.map((c) => [c.day, c.psi])),
  }))
  return drift.score.length ? [score, ...features] : features
}

/** Daily rates for the performance chart (percentages). */
export function performanceSeries(perf: Performance) {
  return perf.days.map((d) => ({
    day: d.day.slice(5),
    Precision: d.precision == null ? null : d.precision * 100,
    Recall: d.recall == null ? null : d.recall * 100,
    'PR-AUC': d.pr_auc == null ? null : d.pr_auc * 100,
    'Fraud $ caught': d.fraud_value_caught == null ? null : d.fraud_value_caught * 100,
  }))
}

/** Daily alert counts for the volume chart. */
export function alertSeries(alerts: Alerts) {
  return alerts.days.map((d) => ({
    day: d.day.slice(5),
    'Model alerts': d.model_alerts,
    'Unusual pattern': d.unusual,
  }))
}
