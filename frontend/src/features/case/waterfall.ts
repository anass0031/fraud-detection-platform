/**
 * SHAP waterfall data. In log-odds the explanation is additive:
 *   base_value (the model's average) + Σ top factors + other factors = raw_log_odds (this case)
 * Each factor becomes a floating bar from the running total to the running total + its
 * contribution. The calibrated probability is shown elsewhere: calibration is applied after
 * the raw score, so it is not part of this sum.
 */
import type { CaseDetail } from './api'

type Prediction = CaseDetail['prediction']
export type Factor = Prediction['top_factors'][number]

export type StepKind = 'raises' | 'lowers' | 'other'

export type WaterfallStep = {
  key: string
  label: string
  text: string
  contribution: number
  start: number
  end: number
  range: [number, number] // [low, high] for the floating bar
  kind: StepKind
  masked: boolean
}

export type Waterfall = {
  base: number
  final: number
  steps: WaterfallStep[]
  domain: [number, number] // x-axis range with a little padding
  complete: boolean // false when the stored explanation lacks the end value
}

const TOLERANCE = 1e-6

export function buildWaterfall(p: Prediction): Waterfall {
  const base = p.base_value ?? 0
  let running = base
  const steps: WaterfallStep[] = []

  const push = (
    key: string,
    label: string,
    text: string,
    contribution: number,
    masked: boolean,
    kind: StepKind,
  ) => {
    const start = running
    const end = running + contribution
    running = end
    steps.push({
      key,
      label,
      text,
      contribution,
      start,
      end,
      range: [Math.min(start, end), Math.max(start, end)],
      kind,
      masked,
    })
  }

  for (const f of p.top_factors) {
    push(
      f.key,
      f.label,
      f.text,
      f.contribution,
      f.masked,
      f.contribution >= 0 ? 'raises' : 'lowers',
    )
  }
  const other = p.other_contribution ?? 0
  if (p.other_contribution !== null && Math.abs(other) > TOLERANCE) {
    push(
      'other',
      'All other factors',
      `All other factors together ${other >= 0 ? 'raise' : 'lower'} the risk`,
      other,
      false,
      'other',
    )
  }

  const final = p.raw_log_odds ?? running
  const complete =
    p.base_value !== null &&
    p.raw_log_odds !== null &&
    p.other_contribution !== null &&
    Math.abs(running - final) < 1e-3

  const values = [base, final, ...steps.flatMap((s) => s.range)]
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const pad = Math.max((hi - lo) * 0.05, 0.1)
  return { base, final, steps, domain: [lo - pad, hi + pad], complete }
}

/** Readable factors first (by size), then the anonymised families (by size). */
export function orderFactors(factors: Factor[]): { readable: Factor[]; masked: Factor[] } {
  const bySize = (a: Factor, b: Factor) => Math.abs(b.contribution) - Math.abs(a.contribution)
  return {
    readable: factors.filter((f) => !f.masked).sort(bySize),
    masked: factors.filter((f) => f.masked).sort(bySize),
  }
}
