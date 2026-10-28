import { screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { caseDetail } from '../../test/fixtures'
import { ADMIN, ANALYST, fakeAuth, renderRoutes } from '../../test/render'
import type { CaseDetail } from './api'
import { CasePage } from './CasePage'

const fetchMock = vi.fn<typeof fetch>()
beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

function serveCase(body: CaseDetail | { detail: string }, status = 200) {
  fetchMock.mockImplementation(async (input) => {
    const path = new URL((input as Request).url).pathname
    if (path === '/api/v1/users') return new Response('[]')
    if (path.endsWith('/timeline')) return new Response('[]')
    return new Response(JSON.stringify(body), { status })
  })
}

const routes = [{ path: '/cases/:caseId', element: <CasePage /> }]

describe('CasePage', () => {
  it('shows the score, the facts and the earlier transactions', async () => {
    serveCase(caseDetail())
    renderRoutes(routes, '/cases/94', fakeAuth(ANALYST))

    expect(await screen.findByRole('heading', { name: 'Case 94' })).toBeInTheDocument()
    expect((fetchMock.mock.calls[0][0] as Request).url).toMatch(/\/api\/v1\/cases\/94$/)
    expect(screen.getByLabelText('Fraud probability')).toHaveTextContent('90%')
    expect(screen.getByText('critical')).toBeInTheDocument()
    expect(screen.getByText('$360.60')).toBeInTheDocument()
    expect(screen.getByText(/Model lgbm-2026\.10\.27-1/)).toBeInTheDocument()
    expect(screen.getByText('Not assigned')).toBeInTheDocument()
    expect(screen.queryByText('Unusual pattern')).not.toBeInTheDocument()

    expect(screen.getByText('2018-04-28 08:37')).toBeInTheDocument()
    expect(screen.getByText('credit')).toBeInTheDocument()
    expect(screen.getByText('mobile')).toBeInTheDocument()

    const history = screen.getByRole('table')
    const rows = within(history).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(2)
    expect(within(rows[0]).getByText('$35.50')).toBeInTheDocument()
    expect(within(rows[1]).getByText('Had a case')).toBeInTheDocument()
    // Analysts can claim a new case; only admins get "Assign".
    expect(screen.getByRole('button', { name: 'Claim' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Assign' })).not.toBeInTheDocument()
  })

  it('explains an unusual-pattern case and shows missing facts as a dash', async () => {
    const c = caseDetail({ reason: 'anomaly', priority_tier: 2 })
    c.prediction = { ...c.prediction, is_alert: false, is_unusual: true }
    c.transaction = { ...c.transaction, device_type: null }
    c.card_history = []
    serveCase(c)
    renderRoutes(routes, '/cases/94', fakeAuth(ADMIN))
    expect(await screen.findByText(/anomaly detector found it/)).toBeInTheDocument()
    expect(screen.getAllByText('Unusual pattern').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('—')).toBeInTheDocument()
    expect(screen.getByText(/No earlier transactions/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Assign' })).toBeInTheDocument()
  })

  it('shows who holds the case and the final decision', async () => {
    serveCase(
      caseDetail({
        status: 'closed',
        assignee: { id: 9, full_name: 'Bob Analyst', email: 'b@x.io' },
        final_decision: 'confirmed_fraud',
      }),
    )
    renderRoutes(routes, '/cases/94', fakeAuth(ANALYST))
    expect(
      await screen.findByText('Assigned to Bob Analyst · Decision: confirmed fraud'),
    ).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Claim' })).not.toBeInTheDocument()
  })

  it('shows the 404 page for an unknown or invalid case id', async () => {
    serveCase({ detail: 'Case not found' }, 404)
    renderRoutes(routes, '/cases/999999', fakeAuth(ANALYST))
    expect(await screen.findByText('This page does not exist.')).toBeInTheDocument()

    renderRoutes(routes, '/cases/abc', fakeAuth(ANALYST))
    expect(screen.getAllByText('This page does not exist.').length).toBe(2)
  })

  it('never shows ground truth', async () => {
    serveCase(caseDetail())
    const { container } = renderRoutes(routes, '/cases/94', fakeAuth(ANALYST))
    await screen.findByRole('heading', { name: 'Case 94' })
    expect(container.textContent).not.toMatch(/isFraud|ground.?truth/i)
  })
})
