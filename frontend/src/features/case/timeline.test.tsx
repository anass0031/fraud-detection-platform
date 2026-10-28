import { screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { caseDetail } from '../../test/fixtures'
import { ANALYST, fakeAuth, renderWith } from '../../test/render'
import type { TimelineEntry } from './api'
import { describeEvent, nameBook } from './timeline'
import { DecisionsList, TimelineCard } from './TimelineCard'

const ANN = { id: 1, full_name: 'Ann Analyst', email: 'a@x.io' }
const ADA = { id: 2, full_name: 'Ada Admin', email: 'ad@x.io' }

function entry(
  id: number,
  action: string,
  after: Record<string, unknown>,
  actor = ANN,
): TimelineEntry {
  return {
    id,
    action,
    actor: action === 'case.create' ? null : actor,
    before: null,
    after,
    comment: null,
    created_at: '2026-10-28T13:00:00Z',
  }
}

const ENTRIES: TimelineEntry[] = [
  entry(1, 'case.create', { reason: 'model_score', priority_score: 360.6, status: 'new' }),
  entry(2, 'case.assign', { status: 'assigned', assigned_to: 1 }, ADA),
  {
    ...entry(3, 'case.decide', { status: 'escalated', decision: 'escalate' }),
    comment: 'Need a second look',
  },
  entry(4, 'case.claim', { status: 'escalated', assigned_to: 2 }, ADA),
  entry(5, 'case.decide', { status: 'closed', decision: 'false_positive' }, ADA),
]

describe('describeEvent', () => {
  const nameOf = nameBook(caseDetail(), ENTRIES, [])

  it('turns audit rows into sentences', () => {
    expect(ENTRIES.map((e) => describeEvent(e, nameOf))).toEqual([
      'Case opened: model alert (expected loss $360.60)',
      'Assigned to Ann Analyst',
      'Escalated to the admins',
      'Claimed',
      'Decision: false positive',
    ])
  })

  it('names an anomaly case and an unknown assignee', () => {
    expect(describeEvent(entry(9, 'case.create', { reason: 'anomaly' }), nameOf)).toBe(
      'Case opened: unusual pattern (anomaly safety net)',
    )
    expect(describeEvent(entry(9, 'case.assign', { assigned_to: 77 }), nameOf)).toBe(
      'Assigned to user #77',
    )
  })
})

describe('TimelineCard', () => {
  beforeEach(() =>
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(ENTRIES))),
    ),
  )
  afterEach(() => vi.unstubAllGlobals())

  it('shows every event with its actor ("System" for scoring) and the comment', async () => {
    renderWith(<TimelineCard c={caseDetail()} />, fakeAuth(ANALYST))
    expect(await screen.findByText('Escalated to the admins')).toBeInTheDocument()
    expect(screen.getByText(/^System ·/)).toBeInTheDocument()
    expect(screen.getAllByText(/^Ada Admin ·/)).toHaveLength(3)
    expect(screen.getByText('“Need a second look”')).toBeInTheDocument()
  })
})

describe('DecisionsList', () => {
  it('lists the decisions with author and comment; nothing when there are none', () => {
    const c = caseDetail({
      decisions: [
        {
          id: 1,
          decision: 'escalate',
          comment: 'Unsure',
          analyst: ANN,
          created_at: '2026-10-28T13:00:00Z',
        },
        {
          id: 2,
          decision: 'false_positive',
          comment: null,
          analyst: ADA,
          created_at: '2026-10-28T14:00:00Z',
        },
      ],
    })
    renderWith(<DecisionsList c={c} />, fakeAuth(ANALYST))
    expect(screen.getByText('Escalated')).toBeInTheDocument()
    expect(screen.getByText('False positive')).toBeInTheDocument()
    expect(screen.getByText('Unsure')).toBeInTheDocument()

    const { container } = renderWith(<DecisionsList c={caseDetail()} />, fakeAuth(ANALYST))
    expect(container.textContent).not.toContain('Decisions')
  })
})
