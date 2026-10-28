import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { caseDetail } from '../../test/fixtures'
import { ANALYST, fakeAuth, renderWith } from '../../test/render'
import { ExplanationCard } from './ExplanationCard'
import { buildWaterfall } from './waterfall'
import { WaterfallChart } from './WaterfallChart'

describe('ExplanationCard', () => {
  it('lists readable reasons first, then the anonymised signals', () => {
    renderWith(<ExplanationCard c={caseDetail()} />, fakeAuth(ANALYST))
    const readable = within(screen.getByRole('list', { name: 'Risk factors' }))
      .getAllByRole('listitem')
      .map((li) => li.textContent)
    expect(readable).toHaveLength(8)
    expect(readable[0]).toContain('Amount $400: strongly raises risk')
    expect(readable.some((t) => t?.includes('Product code is R: moderately lowers risk'))).toBe(
      true,
    )
    const masked = within(screen.getByRole('list', { name: 'Anonymised signals' }))
      .getAllByRole('listitem')
      .map((li) => li.textContent)
    expect(masked).toHaveLength(2)
    expect(masked[0]).toContain('Provider counting signals')
    expect(screen.getByRole('img', { name: 'SHAP waterfall chart' })).toBeInTheDocument()
    expect(screen.queryByText(/Partial explanation/)).not.toBeInTheDocument()
  })

  it('says so when no explanation is stored', () => {
    const c = caseDetail()
    c.prediction = { ...c.prediction, top_factors: [] }
    renderWith(<ExplanationCard c={c} />, fakeAuth(ANALYST))
    expect(screen.getByText('No explanation stored for this prediction.')).toBeInTheDocument()
  })

  it('marks a partial explanation', () => {
    const c = caseDetail()
    c.prediction = { ...c.prediction, raw_log_odds: null }
    renderWith(<ExplanationCard c={c} />, fakeAuth(ANALYST))
    expect(screen.getByText(/Partial explanation/)).toBeInTheDocument()
  })
})

describe('WaterfallChart (rendered SVG)', () => {
  it('draws one floating bar per step, coloured by direction, chained left to right', () => {
    const waterfall = buildWaterfall(caseDetail().prediction)
    const { container } = renderWith(
      <WaterfallChart waterfall={waterfall} fixedWidth={700} />,
      fakeAuth(ANALYST),
    )
    const bars = [...container.querySelectorAll('.recharts-bar-rectangle path')]
    expect(bars).toHaveLength(11)
    const fills = bars.map((b) => b.getAttribute('fill'))
    expect(fills.filter((f) => f?.includes('red'))).toHaveLength(9)
    expect(fills.filter((f) => f?.includes('teal'))).toHaveLength(1)
    expect(fills[10]).toContain('gray')

    // Bars are chained: each one starts where the previous one ended (in pixels).
    const boxes = [...container.querySelectorAll('.recharts-bar-rectangle')].map((g) => {
      const d = g.querySelector('path')!.getAttribute('d')!
      const xs = [...d.matchAll(/[ML]\s*([\d.]+)[ ,]/g)].map((m) => Number(m[1]))
      return { left: Math.min(...xs), right: Math.max(...xs) }
    })
    // Step 1 raises: its right edge is where step 2 starts.
    expect(boxes[1].left).toBeCloseTo(boxes[0].right, 0)
    // Labels and the two reference lines are drawn.
    expect(container.textContent).toContain('Amount')
    expect(container.textContent).toContain('All other factors')
    expect(container.textContent).toContain('Typical')
    expect(container.textContent).toContain('This case')
  })
})
