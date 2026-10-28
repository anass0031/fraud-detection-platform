import { describe, expect, it } from 'vitest'

import { ALERTS, DRIFT, PERFORMANCE, totals } from './fixtures.test-data'
import { alertSeries, heatmapRows, kpis, performanceSeries, psiBand } from './shape'

describe('kpis', () => {
  it('shows the last 7 days and compares with the whole period', () => {
    const cards = kpis(totals({ precision: 0.25, alerts_per_day: 140 }), totals())
    const byKey = Object.fromEntries(cards.map((k) => [k.key, k]))
    expect(byKey.precision.value).toBe('25.0%')
    expect(byKey.precision.previous).toBe('21.5%')
    expect(byKey.precision.delta).toBeCloseTo(0.035)
    expect(byKey.pr_auc.value).toBe('0.356')
    expect(byKey.net_savings.value).toBe('$14,315.00')
    expect(byKey.alerts_per_day.value).toBe('140')
    expect(byKey.alerts_per_day.better).toBe('down')
  })

  it('shows a dash when a value is missing', () => {
    const cards = kpis(totals({ pr_auc: null }), totals())
    const prAuc = cards.find((k) => k.key === 'pr_auc')!
    expect(prAuc.value).toBe('—')
    expect(prAuc.delta).toBeNull()
  })
})

describe('psiBand', () => {
  it.each([
    [0, 'stable'],
    [0.099, 'stable'],
    [0.1, 'moderate'],
    [0.25, 'moderate'],
    [0.26, 'significant'],
  ])('%s -> %s', (psi, band) => {
    expect(psiBand(psi, DRIFT.thresholds)).toBe(band)
  })
})

describe('heatmapRows', () => {
  it('puts the model score first, then features by drift, with readable labels', () => {
    const rows = heatmapRows(DRIFT)
    expect(rows.map((r) => r.label)).toEqual(['Model score', 'Days since D1 (normalised)', 'M3'])
    expect(rows[0].cells.get('2018-04-29')).toBe(0.007)
    expect(rows[2].cells.has('2018-04-28')).toBe(false) // missing day stays empty
  })

  it('keeps only the top features', () => {
    expect(heatmapRows(DRIFT, 1)).toHaveLength(2)
  })
})

describe('chart series', () => {
  it('turns rates into percentages and keeps gaps', () => {
    const s = performanceSeries(PERFORMANCE)
    expect(s[0]).toMatchObject({ day: '04-28', Precision: 20, Recall: 37 })
    expect(s[1]['PR-AUC']).toBeNull()
  })

  it('counts alerts per day', () => {
    expect(alertSeries(ALERTS)[0]).toEqual({
      day: '04-28',
      'Model alerts': 154,
      'Unusual pattern': 2,
    })
  })
})
