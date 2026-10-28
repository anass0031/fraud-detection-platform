import { describe, expect, it } from 'vitest'

import { ADMIN, ANALYST } from '../../test/render'
import { canAssign, canClaim, canDecide, canEscalate } from './permissions'

const ref = (u: { id: number; full_name: string; email: string }) => ({
  id: u.id,
  full_name: u.full_name,
  email: u.email,
})
const OTHER = { id: 99, full_name: 'Other', email: 'o@example.com' }

describe('case permissions', () => {
  it('new case: anyone claims, only admins assign, nobody decides', () => {
    const c = { status: 'new' as const, assignee: null }
    expect([canClaim(ANALYST, c), canClaim(ADMIN, c)]).toEqual([true, true])
    expect([canAssign(ANALYST, c), canAssign(ADMIN, c)]).toEqual([false, true])
    expect(canDecide(ANALYST, c)).toBe(false)
  })

  it('assigned case: only the assignee decides; analysts may escalate', () => {
    const mine = { status: 'assigned' as const, assignee: ref(ANALYST) }
    expect(canClaim(ANALYST, mine)).toBe(false)
    expect(canDecide(ANALYST, mine)).toBe(true)
    expect(canEscalate(ANALYST, mine)).toBe(true)
    const theirs = { status: 'assigned' as const, assignee: OTHER }
    expect(canDecide(ANALYST, theirs)).toBe(false)
    expect(canDecide(ADMIN, theirs)).toBe(false)
    const adminHolds = { status: 'assigned' as const, assignee: ref(ADMIN) }
    expect(canEscalate(ADMIN, adminHolds)).toBe(false)
  })

  it('escalated case: admins only, and it cannot be escalated again', () => {
    const open = { status: 'escalated' as const, assignee: null }
    expect([canClaim(ANALYST, open), canClaim(ADMIN, open)]).toEqual([false, true])
    const held = { status: 'escalated' as const, assignee: ref(ADMIN) }
    expect(canDecide(ADMIN, held)).toBe(true)
    expect(canEscalate(ADMIN, held)).toBe(false)
  })

  it('closed case: read-only', () => {
    const c = { status: 'closed' as const, assignee: ref(ANALYST) }
    expect([canClaim(ANALYST, c), canAssign(ADMIN, c), canDecide(ANALYST, c)]).toEqual([
      false,
      false,
      false,
    ])
  })
})
