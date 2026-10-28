/**
 * The champion vs challenger block stored with a retrained model (`metrics.comparison`,
 * written by ml/scripts/retrain.py). It arrives as free-form JSON, so it is checked here.
 */
import { money } from '../../utils/format'

export type WindowMetrics = {
  n: number
  n_fraud: number
  alerts_per_day: number
  precision: number | null
  recall: number | null
  pr_auc: number | null
  fraud_value_caught: number | null
  net_savings: number
}

type Side = { overall: WindowMetrics; excl_fraud_customers: WindowMetrics }

export type Comparison = {
  window: [string, string]
  champion_version: string
  champion_threshold: number
  challenger_threshold: number
  champion: Side
  challenger: Side
  pr_auc_diff_bootstrap?: {
    diff: number
    low: number
    high: number
    share_positive: number
    days: number
  }
}

export type Scope = keyof Side

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null

/** The comparison of a model, or null when it has none (e.g. the original model). */
export function readComparison(metrics: unknown): Comparison | null {
  if (!isObj(metrics) || !isObj(metrics.comparison)) return null
  const c = metrics.comparison
  const sideOk = (s: unknown) =>
    isObj(s) && isObj(s.overall) && isObj(s.excl_fraud_customers) && 'pr_auc' in s.overall
  if (!sideOk(c.champion) || !sideOk(c.challenger) || !Array.isArray(c.window)) return null
  return c as unknown as Comparison
}

export type Row = {
  key: keyof WindowMetrics
  label: string
  champion: string
  challenger: string
  better: boolean | null // challenger better than champion?
}

const pct = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const ROWS: {
  key: keyof WindowMetrics
  label: string
  fmt: (v: number | null) => string
  up: boolean | null // null: neither direction is "better" (shown, not coloured)
}[] = [
  { key: 'pr_auc', label: 'PR-AUC', fmt: (v) => (v == null ? '—' : v.toFixed(3)), up: true },
  { key: 'precision', label: 'Precision of alerts', fmt: pct, up: true },
  { key: 'recall', label: 'Recall', fmt: pct, up: true },
  { key: 'fraud_value_caught', label: 'Fraud $ caught', fmt: pct, up: true },
  { key: 'net_savings', label: 'Net savings', fmt: (v) => (v == null ? '—' : money(v)), up: true },
  {
    key: 'alerts_per_day',
    label: 'Alerts per day',
    fmt: (v) => (v == null ? '—' : v.toFixed(1)),
    // Fewer alerts is less work but can also mean missed fraud: not coloured.
    up: null,
  },
]

/** Side-by-side rows; `better` says whether the challenger improves on the champion. */
export function comparisonRows(c: Comparison, scope: Scope): Row[] {
  const a = c.champion[scope]
  const b = c.challenger[scope]
  return ROWS.map(({ key, label, fmt, up }) => {
    const x = a[key] as number | null
    const y = b[key] as number | null
    return {
      key,
      label,
      champion: fmt(x),
      challenger: fmt(y),
      better: up === null || x == null || y == null || x === y ? null : up ? y > x : y < x,
    }
  })
}

/** One sentence about how sure the PR-AUC gain is (day-block bootstrap). */
export function confidenceText(c: Comparison): string | null {
  const b = c.pr_auc_diff_bootstrap
  if (!b) return null
  const sign = (v: number) => (v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3))
  const verdict =
    b.low > 0
      ? 'the challenger is better beyond chance'
      : b.high < 0
        ? 'the challenger is worse beyond chance'
        : 'the difference may be chance'
  return (
    `PR-AUC difference ${sign(b.diff)}, 95% interval [${sign(b.low)}, ${sign(b.high)}] ` +
    `(resampling whole days, ${b.days} days): ${verdict}.`
  )
}
