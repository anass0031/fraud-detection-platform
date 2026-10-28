import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { caseDetail } from '../../test/fixtures'
import { ADMIN, ANALYST, fakeAuth, renderWith } from '../../test/render'
import type { CaseDetail } from './api'
import { DecisionPanel } from './DecisionPanel'

const ME = { id: ANALYST.id, full_name: ANALYST.full_name, email: ANALYST.email }
const ADMIN_REF = { id: ADMIN.id, full_name: ADMIN.full_name, email: ADMIN.email }

const fetchMock = vi.fn<typeof fetch>()
const posted: unknown[] = []
beforeEach(() => {
  posted.length = 0
  fetchMock.mockReset()
  fetchMock.mockImplementation(async (input) => {
    const req = input as Request
    if (req.method === 'POST') {
      const body = (await req.json()) as { decision: string }
      posted.push(body)
      const status = body.decision === 'escalate' ? 'escalated' : 'closed'
      return new Response(JSON.stringify(caseDetail({ status })), { status: 200 })
    }
    return new Response('{}')
  })
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

const mine = (over: Partial<CaseDetail> = {}) =>
  caseDetail({ status: 'assigned', assignee: ME, ...over })

describe('DecisionPanel: who sees the form', () => {
  it.each([
    [
      'closed case',
      caseDetail({ status: 'closed', assignee: ME }),
      ANALYST,
      'This case is closed.',
    ],
    ['unclaimed case', caseDetail(), ANALYST, 'Claim the case to record a decision.'],
    [
      "someone else's case",
      caseDetail({ status: 'assigned', assignee: { ...ME, id: 42, full_name: 'Bob' } }),
      ANALYST,
      'Assigned to Bob.',
    ],
    [
      'escalated case (analyst)',
      caseDetail({ status: 'escalated', assignee: null }),
      ANALYST,
      'Escalated: waiting for an admin to take a decision.',
    ],
  ])('%s: explains why there is no form', (_name, c, user, message) => {
    renderWith(<DecisionPanel c={c} />, fakeAuth(user))
    expect(screen.getByText(message)).toBeInTheDocument()
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
  })

  it('the assigned analyst gets three options', () => {
    renderWith(<DecisionPanel c={mine()} />, fakeAuth(ANALYST))
    expect(screen.getAllByRole('radio')).toHaveLength(3)
    expect(screen.getByRole('radio', { name: /Escalated/ })).toBeInTheDocument()
  })

  it('an admin deciding an escalated case cannot escalate again', () => {
    const c = caseDetail({ status: 'escalated', assignee: ADMIN_REF })
    renderWith(<DecisionPanel c={c} />, fakeAuth(ADMIN))
    expect(screen.getAllByRole('radio')).toHaveLength(2)
    expect(screen.getByText(/An analyst escalated this case/)).toBeInTheDocument()
  })
})

describe('DecisionPanel: recording a decision', () => {
  it('asks for confirmation before closing, then sends the decision and comment', async () => {
    const user = userEvent.setup()
    renderWith(<DecisionPanel c={mine()} />, fakeAuth(ANALYST))
    const record = screen.getByRole('button', { name: 'Record decision' })
    expect(record).toBeDisabled()

    await user.click(screen.getByRole('radio', { name: /Confirmed fraud/ }))
    await user.type(screen.getByRole('textbox', { name: /Comment/ }), '  Card used abroad  ')
    await user.click(record)
    expect(await screen.findByText('Close case 94?')).toBeInTheDocument()
    expect(posted).toHaveLength(0) // nothing sent before confirming

    await user.click(screen.getByRole('button', { name: 'Close case' }))
    await waitFor(() =>
      expect(posted).toEqual([{ decision: 'confirmed_fraud', comment: 'Card used abroad' }]),
    )
    expect(await screen.findByText('Case 94 closed: confirmed fraud')).toBeInTheDocument()
  })

  it('cancel keeps the case open', async () => {
    const user = userEvent.setup()
    renderWith(<DecisionPanel c={mine()} />, fakeAuth(ANALYST))
    await user.click(screen.getByRole('radio', { name: /False positive/ }))
    await user.click(screen.getByRole('button', { name: 'Record decision' }))
    await user.click(await screen.findByRole('button', { name: 'Cancel' }))
    expect(posted).toHaveLength(0)
  })

  it('escalating needs a comment and is sent without the closing dialog', async () => {
    const user = userEvent.setup()
    renderWith(<DecisionPanel c={mine()} />, fakeAuth(ANALYST))
    await user.click(screen.getByRole('radio', { name: /Escalated/ }))
    const escalate = screen.getByRole('button', { name: 'Escalate' })
    expect(escalate).toBeDisabled()
    await user.type(screen.getByRole('textbox', { name: /Comment/ }), '   ')
    expect(escalate).toBeDisabled()
    expect(screen.getByText('A comment is required to escalate')).toBeInTheDocument()
    await user.type(screen.getByRole('textbox', { name: /Comment/ }), 'Unusual merchant')
    await user.click(escalate)
    await waitFor(() =>
      expect(posted).toEqual([{ decision: 'escalate', comment: 'Unusual merchant' }]),
    )
    expect(await screen.findByText('Case 94 escalated to the admins')).toBeInTheDocument()
  })

  it('shows the server message if the decision is refused', async () => {
    fetchMock.mockImplementation(
      async () => new Response(JSON.stringify({ detail: 'Case is closed' }), { status: 409 }),
    )
    const user = userEvent.setup()
    renderWith(<DecisionPanel c={mine()} />, fakeAuth(ANALYST))
    await user.click(screen.getByRole('radio', { name: /False positive/ }))
    await user.click(screen.getByRole('button', { name: 'Record decision' }))
    await user.click(await screen.findByRole('button', { name: 'Close case' }))
    expect(await screen.findByText('Case is closed')).toBeInTheDocument()
  })
})
