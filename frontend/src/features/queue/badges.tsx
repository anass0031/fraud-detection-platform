import { Badge } from '@mantine/core'

import type { Schemas } from '../../api/client'

type Risk = Schemas['CaseListItem']['risk_level']
type Status = Schemas['CaseListItem']['status']
type Reason = Schemas['CaseListItem']['reason']

const RISK_COLOR: Record<Risk, string> = {
  low: 'gray',
  medium: 'yellow',
  high: 'orange',
  critical: 'red',
}

const STATUS: Record<Status, { label: string; color: string }> = {
  new: { label: 'New', color: 'blue' },
  assigned: { label: 'Assigned', color: 'indigo' },
  escalated: { label: 'Escalated', color: 'grape' },
  closed: { label: 'Closed', color: 'gray' },
}

export function RiskBadge({ level }: { level: Risk }) {
  return (
    <Badge color={RISK_COLOR[level]} variant={level === 'critical' ? 'filled' : 'light'}>
      {level}
    </Badge>
  )
}

export function StatusBadge({ status }: { status: Status }) {
  const { label, color } = STATUS[status]
  return (
    <Badge color={color} variant="outline">
      {label}
    </Badge>
  )
}

/** Model alerts need no badge; the anomaly safety net is called out. */
export function ReasonBadge({ reason }: { reason: Reason }) {
  return reason === 'anomaly' ? (
    <Badge color="teal" variant="light">
      Unusual pattern
    </Badge>
  ) : (
    <Badge color="gray" variant="transparent">
      Model score
    </Badge>
  )
}
