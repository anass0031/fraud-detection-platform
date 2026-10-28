import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ANALYST, fakeAuth, renderWith } from '../../test/render'
import { ALERTS, DRIFT, FEEDBACK, PERFORMANCE } from './fixtures.test-data'
import { MonitoringPage } from './MonitoringPage'

const ANSWERS: Record<string, unknown> = {
  performance: PERFORMANCE,
  drift: DRIFT,
  alerts: ALERTS,
  feedback: FEEDBACK,
}

const fetchMock = vi.fn<typeof fetch>()
beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

function serve(answers: Record<string, unknown>, status = 200) {
  fetchMock.mockImplementation(async (input) => {
    const name = new URL((input as Request).url).pathname.split('/').pop()!
    return new Response(JSON.stringify(answers[name] ?? { detail: 'unknown' }), { status })
  })
}

describe('MonitoringPage', () => {
  it('shows KPIs for the last 7 days, the drift heatmap, backlog and feedback', async () => {
    serve(ANSWERS)
    renderWith(<MonitoringPage />, fakeAuth(ANALYST))

    expect(await screen.findByLabelText('Precision of alerts last 7 days')).toHaveTextContent(
      '25.0%',
    )
    expect(screen.getByText(/Model lgbm-2026\.10\.27-1 on 3 replayed days/)).toBeInTheDocument()
    expect(screen.getByLabelText('Net savings last 7 days')).toHaveTextContent('$14,315.00')

    const heatmap = screen.getByRole('table', { name: 'Drift heatmap' })
    const rows = within(heatmap).getAllByRole('row').slice(1)
    expect(rows.map((r) => r.firstChild?.textContent)).toEqual([
      'Model score',
      'Days since D1 (normalised)',
      'M3',
    ])
    const bands = [...heatmap.querySelectorAll('[data-band]')].map((c) =>
      c.getAttribute('data-band'),
    )
    // score stable x2, D1n significant x2, M3: missing then moderate
    expect(bands).toEqual(['stable', 'stable', 'significant', 'significant', 'none', 'moderate'])

    expect(screen.getByLabelText('Open backlog')).toHaveTextContent('441')
    expect(screen.getByText('2 confirmed fraud')).toBeInTheDocument()
    expect(screen.getByText('1 of 3')).toBeInTheDocument()
  })

  it('explains what to do when nothing is replayed yet', async () => {
    serve({
      ...ANSWERS,
      performance: { model_version: 'm', days: [], total: null, last_7_days: null },
    })
    renderWith(<MonitoringPage />, fakeAuth(ANALYST))
    expect(await screen.findByText(/No replayed day yet/)).toBeInTheDocument()
  })

  it('shows an error with a retry', async () => {
    fetchMock.mockImplementation(
      async () => new Response(JSON.stringify({ detail: 'Database unavailable' }), { status: 503 }),
    )
    renderWith(<MonitoringPage />, fakeAuth(ANALYST))
    expect(await screen.findByText('Database unavailable')).toBeInTheDocument()
    serve(ANSWERS)
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByLabelText('Open backlog')).toBeInTheDocument()
  })

  it('never shows ground truth', async () => {
    serve(ANSWERS)
    const { container } = renderWith(<MonitoringPage />, fakeAuth(ANALYST))
    await screen.findByLabelText('Open backlog')
    expect(container.textContent).not.toMatch(/isFraud|ground.?truth/i)
  })
})
