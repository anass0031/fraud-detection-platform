/** Case timeline: audit rows -> readable sentences. */
import { money } from '../../utils/format'
import { type CaseDetail, DECISION_LABEL, type DecisionKind, type TimelineEntry } from './api'

/** User ids -> names from everything the page already knows (no extra API for analysts). */
export function nameBook(
  c: CaseDetail,
  entries: TimelineEntry[],
  extra: { id: number; full_name: string }[],
) {
  const names = new Map<number, string>()
  extra.forEach((u) => names.set(u.id, u.full_name))
  entries.forEach((e) => e.actor && names.set(e.actor.id, e.actor.full_name))
  c.decisions.forEach((d) => names.set(d.analyst.id, d.analyst.full_name))
  if (c.assignee) names.set(c.assignee.id, c.assignee.full_name)
  return (id: unknown) => (typeof id === 'number' ? (names.get(id) ?? `user #${id}`) : 'nobody')
}

/** One audit row as a sentence. */
export function describeEvent(e: TimelineEntry, nameOf: (id: unknown) => string): string {
  const after = e.after ?? {}
  switch (e.action) {
    case 'case.create':
      return after.reason === 'anomaly'
        ? 'Case opened: unusual pattern (anomaly safety net)'
        : `Case opened: model alert (expected loss ${money(Number(after.priority_score ?? 0))})`
    case 'case.claim':
      return 'Claimed'
    case 'case.assign':
      return `Assigned to ${nameOf(after.assigned_to)}`
    case 'case.decide': {
      const label = DECISION_LABEL[after.decision as DecisionKind] ?? String(after.decision)
      return after.decision === 'escalate'
        ? 'Escalated to the admins'
        : `Decision: ${label.toLowerCase()}`
    }
    default:
      return e.action
  }
}
