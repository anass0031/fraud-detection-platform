import { Alert, Badge, Loader, Paper, Stack, Text, Timeline, Title } from '@mantine/core'

import { errorMessage } from '../../api/client'
import { localTime } from '../../utils/format'
import { useAuth } from '../auth/context'
import { useActiveUsers } from '../queue/api'
import { type CaseDetail, DECISION_LABEL, type TimelineEntry, useTimeline } from './api'
import { describeEvent, nameBook } from './timeline'

const DOT: Record<string, string> = {
  'case.create': 'gray',
  'case.claim': 'blue',
  'case.assign': 'indigo',
  'case.decide': 'grape',
}

function Events({ c, entries }: { c: CaseDetail; entries: TimelineEntry[] }) {
  const { user } = useAuth()
  const users = useActiveUsers(user?.role === 'admin')
  const nameOf = nameBook(c, entries, users.data ?? [])
  return (
    <Timeline bulletSize={14} lineWidth={2} active={entries.length - 1}>
      {entries.map((e) => (
        <Timeline.Item
          key={e.id}
          color={DOT[e.action] ?? 'gray'}
          title={<Text size="sm">{describeEvent(e, nameOf)}</Text>}
        >
          <Text size="xs" c="dimmed">
            {e.actor ? e.actor.full_name : 'System'} · {localTime(e.created_at)}
          </Text>
          {e.comment && (
            <Text size="sm" mt={4} fs="italic">
              “{e.comment}”
            </Text>
          )}
        </Timeline.Item>
      ))}
    </Timeline>
  )
}

/** Everything that happened to the case, oldest first (from the audit trail). */
export function TimelineCard({ c }: { c: CaseDetail }) {
  const timeline = useTimeline(c.id)
  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <Title order={4}>Timeline</Title>
        {timeline.isPending && <Loader size="sm" aria-label="Loading timeline" />}
        {timeline.isError && (
          <Alert color="red" variant="light">
            {errorMessage(timeline.error)}
          </Alert>
        )}
        {timeline.data && <Events c={c} entries={timeline.data} />}
      </Stack>
    </Paper>
  )
}

const DECISION_COLOR = { confirmed_fraud: 'red', false_positive: 'teal', escalate: 'grape' }

/** The decisions so far (an escalation chain can have several). */
export function DecisionsList({ c }: { c: CaseDetail }) {
  if (c.decisions.length === 0) return null
  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <Title order={4}>Decisions</Title>
        {c.decisions.map((d) => (
          <div key={d.id}>
            <Badge variant="light" color={DECISION_COLOR[d.decision]}>
              {DECISION_LABEL[d.decision]}
            </Badge>
            <Text size="xs" c="dimmed" mt={4}>
              {d.analyst.full_name} · {localTime(d.created_at)}
            </Text>
            {d.comment && <Text size="sm">{d.comment}</Text>}
          </div>
        ))}
      </Stack>
    </Paper>
  )
}
