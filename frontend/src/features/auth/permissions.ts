import type { Schemas, User } from '../../api/client'

/** The case fields the rules need (both list items and details have them). */
type CaseLike = Pick<Schemas['CaseListItem'], 'status' | 'assignee'>

/*
 * Same rules as the API (the API stays the authority; these only decide what the UI offers).
 *  new        -> anyone can claim; admin can assign
 *  assigned   -> the assignee decides (analysts may also escalate); admin can reassign
 *  escalated  -> admins only: claim/assign (to an admin) when unassigned, then decide
 *  closed     -> read-only
 */

export function canClaim(user: User, c: CaseLike): boolean {
  if (c.assignee !== null) return false
  if (c.status === 'new') return true
  return c.status === 'escalated' && user.role === 'admin'
}

export function canAssign(user: User, c: CaseLike): boolean {
  return user.role === 'admin' && c.status !== 'closed'
}

export function canDecide(user: User, c: CaseLike): boolean {
  return (c.status === 'assigned' || c.status === 'escalated') && c.assignee?.id === user.id
}

export function canEscalate(user: User, c: CaseLike): boolean {
  return canDecide(user, c) && c.status === 'assigned' && user.role === 'analyst'
}

export function isAdmin(user: User | null): boolean {
  return user?.role === 'admin'
}
