import { describe, expect, it } from 'vitest'

import { FACTORS_94, caseDetail } from '../../test/fixtures'
import { buildWaterfall, orderFactors } from './waterfall'

const p94 = caseDetail().prediction

describe('buildWaterfall', () => {
  it('chains the bars from the model average to this case and adds up exactly', () => {
    const w = buildWaterfall(p94)
    expect(w.base).toBe(-4.774)
    expect(w.final).toBe(2.936)
    expect(w.complete).toBe(true)
    // 10 factors + "all other factors"
    expect(w.steps).toHaveLength(11)
    expect(w.steps[0].start).toBe(-4.774)
    for (let i = 1; i < w.steps.length; i++) {
      expect(w.steps[i].start).toBeCloseTo(w.steps[i - 1].end, 10)
    }
    expect(w.steps[w.steps.length - 1].end).toBeCloseTo(w.final, 3)
  })

  it('colours by direction and keeps floating ranges ordered low -> high', () => {
    const w = buildWaterfall(p94)
    const product = w.steps.find((s) => s.key === 'ProductCD')!
    expect(product.kind).toBe('lowers')
    expect(product.range[0]).toBeLessThan(product.range[1])
    expect(product.range[1]).toBeCloseTo(product.start, 10)
    expect(w.steps[0].kind).toBe('raises')
    expect(w.steps[w.steps.length - 1]).toMatchObject({ key: 'other', kind: 'other' })
  })

  it('pads the axis around every value', () => {
    const w = buildWaterfall(p94)
    expect(w.domain[0]).toBeLessThan(-4.774)
    expect(w.domain[1]).toBeGreaterThan(Math.max(...w.steps.map((s) => s.end)))
  })

  it('handles an explanation without the end value (older predictions)', () => {
    const w = buildWaterfall({ ...p94, raw_log_odds: null, other_contribution: null })
    expect(w.complete).toBe(false)
    expect(w.steps).toHaveLength(10)
    expect(w.final).toBeCloseTo(w.steps[9].end, 10)
  })

  it('skips a negligible "other factors" bar', () => {
    const w = buildWaterfall({ ...p94, other_contribution: 0 })
    expect(w.steps.some((s) => s.key === 'other')).toBe(false)
  })
})

describe('orderFactors', () => {
  it('puts readable factors first, each group by size', () => {
    const { readable, masked } = orderFactors(FACTORS_94)
    expect(masked.map((f) => f.key)).toEqual(['masked_counts', 'masked_timing'])
    expect(readable[0].key).toBe('amt_log')
    expect(readable.map((f) => Math.abs(f.contribution))).toEqual(
      [...readable.map((f) => Math.abs(f.contribution))].sort((a, b) => b - a),
    )
    expect(readable).toHaveLength(8)
  })
})
