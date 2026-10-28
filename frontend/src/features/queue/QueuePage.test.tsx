import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { caseRow } from '../../test/fixtures'
import { ADMIN, ANALYST, fakeAuth, renderRoutes } from '../../test/render'
import type { CaseRow } from './api'
import { QueuePage } from './QueuePage'

const fetchMock = vi.fn<typeof fetch>()
beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status })
}

function page(items: CaseRow[], total = items.length) {
  return json({ items, total, limit: 50, offset: 0 })
}

/** Answer GET /cases with `items` and record every request. */
function serve(items: CaseRow[], total = items.length, extra?: (req: Request) => Response | null) {
  fetchMock.mockImplementation(async (input) => {
    const req = input as Request
    const answer = extra?.(req)
    if (answer) return answer
    if (new URL(req.url).pathname === '/api/v1/cases') return page(items, total)
    return json({ detail: 'unexpected' }, 500)
  })
}

/** The query params of the last GET /cases request. */
function lastQuery(): URLSearchParams {
  const calls = fetchMock.mock.calls
    .map(([input]) => new URL((input as Request).url))
    .filter((url) => url.pathname === '/api/v1/cases')
  return calls[calls.length - 1].searchParams
}

const routes = [
  { path: '/queue', element: <QueuePage /> },
  { path: '/cases/:id', element: <p>case page</p> },
]

describe('QueuePage', () => {
  it('shows open cases in priority order', async () => {
    serve(
      [
        caseRow(),
        caseRow({
          id: 7,
          reason: 'anomaly',
          priority_tier: 2,
          risk_level: 'medium',
          fraud_score: 0.003,
          expected_loss: 1.38,
        }),
      ],
      156,
    )
    renderRoutes(routes, '/queue', fakeAuth(ANALYST))

    const first = await screen.findByRole('row', { name: 'Case 122' })
    expect(within(first).getByText('high')).toBeInTheDocument()
    expect(within(first).getByText('48%')).toBeInTheDocument()
    expect(within(first).getByText('$480.89')).toBeInTheDocument()
    expect(within(first).getByText('$994.00')).toBeInTheDocument()
    expect(within(first).getByText('2018-04-28 19:52')).toBeInTheDocument()
    expect(within(first).getByText('Unassigned')).toBeInTheDocument()
    const second = screen.getByRole('row', { name: 'Case 7' })
    expect(within(second).getByText('Unusual pattern')).toBeInTheDocument()
    expect(within(second).getByText('<1%')).toBeInTheDocument()
    expect(screen.getByText('156 cases')).toBeInTheDocument()

    const q = lastQuery()
    expect(q.getAll('status')).toEqual(['new'])
    expect([q.get('sort'), q.get('order'), q.get('limit'), q.get('offset')]).toEqual([
      'priority',
      'desc',
      '50',
      '0',
    ])
  })

  it('opens the case when a row is clicked', async () => {
    serve([caseRow()])
    const { router } = renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    await userEvent.click(await screen.findByRole('row', { name: 'Case 122' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/cases/122'))
  })

  it('shows the error and retries', async () => {
    fetchMock.mockResolvedValueOnce(json({ detail: 'Database unavailable' }, 503))
    renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    expect(await screen.findByText('Database unavailable')).toBeInTheDocument()
    serve([caseRow()])
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByRole('row', { name: 'Case 122' })).toBeInTheDocument()
  })

  it('"My cases" tab asks for my open cases and is kept in the URL', async () => {
    serve([])
    const { router } = renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    await userEvent.click(await screen.findByRole('tab', { name: 'My cases' }))
    await waitFor(() => expect(lastQuery().get('assignee')).toBe('me'))
    expect(lastQuery().getAll('status')).toEqual(['assigned', 'escalated'])
    expect(router.state.location.search).toBe('?tab=mine')
    expect(await screen.findByText(/You hold no open cases/)).toBeInTheDocument()
  })

  it('reads filters from the URL and clears them', async () => {
    serve([])
    const { router } = renderRoutes(routes, '/queue?risk=critical&min=100', fakeAuth(ANALYST))
    await waitFor(() => expect(lastQuery().getAll('risk_level')).toEqual(['critical']))
    expect(lastQuery().get('amount_min')).toBe('100')
    expect(await screen.findByText('No cases match these filters.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Clear filters' }))
    await waitFor(() => expect(router.state.location.search).toBe(''))
    expect(lastQuery().get('risk_level')).toBeNull()
  })

  it('sorts by clicking a header: descending first, then ascending', async () => {
    serve([caseRow()])
    renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    const amount = await screen.findByRole('button', { name: /Amount/ })
    await userEvent.click(amount)
    await waitFor(() => expect(lastQuery().get('sort')).toBe('amount'))
    expect(lastQuery().get('order')).toBe('desc')
    await userEvent.click(screen.getByRole('button', { name: /Amount/ }))
    await waitFor(() => expect(lastQuery().get('order')).toBe('asc'))
  })

  it('pages through results', async () => {
    serve([caseRow()], 120)
    renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    await userEvent.click(await screen.findByRole('button', { name: '2' }))
    await waitFor(() => expect(lastQuery().get('offset')).toBe('50'))
  })
})

describe('claim and assign', () => {
  it('an analyst claims a case; the queue is refreshed', async () => {
    serve([caseRow()], 1, (req) =>
      req.method === 'POST' && req.url.endsWith('/api/v1/cases/122/claim')
        ? json(caseRow({ status: 'assigned', assignee: { id: 1, full_name: 'Ann', email: 'a' } }))
        : null,
    )
    renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    const row = await screen.findByRole('row', { name: 'Case 122' })
    expect(within(row).queryByRole('button', { name: 'Assign' })).not.toBeInTheDocument()
    const before = fetchMock.mock.calls.length
    await userEvent.click(within(row).getByRole('button', { name: 'Claim' }))
    expect(await screen.findByText('Case 122 is now yours')).toBeInTheDocument()
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(before + 1))
  })

  it('shows the server message when someone else claimed it first', async () => {
    serve([caseRow()], 1, (req) =>
      req.method === 'POST'
        ? json({ detail: 'Case is already assigned to someone else' }, 409)
        : null,
    )
    renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    const row = await screen.findByRole('row', { name: 'Case 122' })
    await userEvent.click(within(row).getByRole('button', { name: 'Claim' }))
    expect(await screen.findByText('Case is already assigned to someone else')).toBeInTheDocument()
  })

  it('an admin assigns a case from the menu', async () => {
    const users = [
      { ...ANALYST, id: 5, full_name: 'Bob Analyst' },
      { ...ANALYST, id: 6, full_name: 'Gone User', is_active: false },
    ]
    let body: unknown = null
    serve([caseRow()], 1, (req) => {
      const path = new URL(req.url).pathname
      if (path === '/api/v1/users') return json(users)
      if (path === '/api/v1/cases/122/assign') {
        void req.json().then((b) => (body = b))
        return json(
          caseRow({
            status: 'assigned',
            assignee: { id: 5, full_name: 'Bob Analyst', email: 'b' },
          }),
        )
      }
      return null
    })
    renderRoutes(routes, '/queue', fakeAuth(ADMIN))
    const row = await screen.findByRole('row', { name: 'Case 122' })
    await userEvent.click(within(row).getByRole('button', { name: 'Assign' }))
    expect(await screen.findByRole('menuitem', { name: 'Bob Analyst' })).toBeInTheDocument()
    expect(screen.queryByRole('menuitem', { name: 'Gone User' })).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('menuitem', { name: 'Bob Analyst' }))
    expect(await screen.findByText('Case 122 assigned to Bob Analyst')).toBeInTheDocument()
    expect(body).toEqual({ user_id: 5 })
  })
})
