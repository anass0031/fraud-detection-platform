import { describe, expect, it } from 'vitest'

import { comparisonRows, confidenceText, readComparison } from './comparison'
import { COMPARISON } from './models.test-data'

describe('readComparison', () => {
  it('reads a stored comparison', () => {
    expect(readComparison({ comparison: COMPARISON })?.champion_version).toBe('lgbm-2026.10.27-1')
  })

  it.each([null, {}, { comparison: null }, { comparison: { champion: {} } }])(
    'returns null for %j',
    (metrics) => {
      expect(readComparison(metrics)).toBeNull()
    },
  )
})

describe('comparisonRows', () => {
  it('formats both sides and says where the challenger is better', () => {
    const rows = Object.fromEntries(comparisonRows(COMPARISON, 'overall').map((r) => [r.key, r]))
    expect(rows.pr_auc).toMatchObject({ champion: '0.432', challenger: '0.532', better: true })
    expect(rows.precision).toMatchObject({ champion: '25.7%', challenger: '35.2%', better: true })
    expect(rows.net_savings.challenger).toBe('$72,307.00')
    // Fewer alerts is not automatically better: shown without colour.
    expect(rows.alerts_per_day).toMatchObject({ challenger: '136.0', better: null })
  })

  it('switches to the excluded-customers view', () => {
    const rows = comparisonRows(COMPARISON, 'excl_fraud_customers')
    expect(rows[0]).toMatchObject({ champion: '0.454', challenger: '0.519' })
  })

  it('marks a worse challenger', () => {
    const worse = {
      ...COMPARISON,
      challenger: {
        ...COMPARISON.challenger,
        overall: { ...COMPARISON.champion.overall, pr_auc: 0.4 },
      },
    }
    expect(comparisonRows(worse, 'overall')[0].better).toBe(false)
  })
})

describe('confidenceText', () => {
  it('states the interval and the verdict', () => {
    expect(confidenceText(COMPARISON)).toBe(
      'PR-AUC difference +0.100, 95% interval [+0.064, +0.140] (resampling whole days, 7 days): ' +
        'the challenger is better beyond chance.',
    )
    const unsure = {
      ...COMPARISON,
      pr_auc_diff_bootstrap: { diff: 0.01, low: -0.02, high: 0.04, share_positive: 0.7, days: 7 },
    }
    expect(confidenceText(unsure)).toContain('the difference may be chance')
    expect(confidenceText({ ...COMPARISON, pr_auc_diff_bootstrap: undefined })).toBeNull()
  })
})
