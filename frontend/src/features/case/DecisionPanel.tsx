import {
  Alert,
  Button,
  Group,
  Modal,
  Paper,
  Radio,
  Stack,
  Text,
  Textarea,
  Title,
} from '@mantine/core'
import { useState } from 'react'

import { useAuth } from '../auth/context'
import { useCan } from '../auth/useCan'
import { type CaseDetail, DECISION_LABEL, type DecisionKind, useDecide } from './api'

const MAX_COMMENT = 2000

const OPTION_HELP: Record<DecisionKind, string> = {
  confirmed_fraud: 'The transaction is fraudulent. Closes the case.',
  false_positive: 'The transaction is legitimate. Closes the case.',
  escalate: 'Not sure: hand the case to an admin. A comment is required.',
}

/** Why the user cannot decide right now (shown instead of the form). */
function notAllowedReason(c: CaseDetail, userId: number, isAdmin: boolean): string {
  if (c.status === 'closed') return 'This case is closed.'
  if (c.status === 'escalated' && !isAdmin)
    return 'Escalated: waiting for an admin to take a decision.'
  if (c.assignee === null) return 'Claim the case to record a decision.'
  if (c.assignee.id !== userId) return `Assigned to ${c.assignee.full_name}.`
  return 'You cannot decide on this case.'
}

export function DecisionPanel({ c }: { c: CaseDetail }) {
  const { user } = useAuth()
  const can = useCan()
  const decide = useDecide()
  const [decision, setDecision] = useState<DecisionKind | null>(null)
  const [comment, setComment] = useState('')
  const [confirming, setConfirming] = useState(false)

  if (!user) return null
  const allowed = can.decide(c)
  const options: DecisionKind[] = can.escalate(c)
    ? ['confirmed_fraud', 'false_positive', 'escalate']
    : ['confirmed_fraud', 'false_positive']

  const trimmed = comment.trim()
  const needsComment = decision === 'escalate' && trimmed === ''
  const tooLong = comment.length > MAX_COMMENT
  const ready = decision !== null && !needsComment && !tooLong

  const submit = () => {
    if (!decision) return
    decide.mutate(
      { caseId: c.id, decision, comment: trimmed || null },
      {
        onSuccess: () => {
          setDecision(null)
          setComment('')
        },
        onSettled: () => setConfirming(false),
      },
    )
  }

  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <Title order={4}>Decision</Title>
        {!allowed ? (
          <Text size="sm" c="dimmed">
            {notAllowedReason(c, user.id, user.role === 'admin')}
          </Text>
        ) : (
          <>
            {c.status === 'escalated' && (
              <Alert color="grape" variant="light">
                An analyst escalated this case. Read their comment below, then confirm fraud or mark
                it as a false positive.
              </Alert>
            )}
            <Radio.Group
              value={decision}
              onChange={(v) => setDecision(v as DecisionKind)}
              label="Your decision"
            >
              <Stack gap="xs" mt="xs">
                {options.map((o) => (
                  <Radio key={o} value={o} label={DECISION_LABEL[o]} description={OPTION_HELP[o]} />
                ))}
              </Stack>
            </Radio.Group>
            <Textarea
              label="Comment"
              description={
                decision === 'escalate' ? 'Required: what should the admin look at?' : 'Optional'
              }
              autosize
              minRows={2}
              maxRows={6}
              value={comment}
              onChange={(e) => setComment(e.currentTarget.value)}
              error={
                tooLong
                  ? `At most ${MAX_COMMENT} characters`
                  : needsComment && comment.length > 0
                    ? 'A comment is required to escalate'
                    : undefined
              }
            />
            <Group justify="space-between">
              <Text size="xs" c={tooLong ? 'red' : 'dimmed'}>
                {comment.length} / {MAX_COMMENT}
              </Text>
              <Button
                disabled={!ready}
                loading={decide.isPending}
                onClick={() => (decision === 'escalate' ? submit() : setConfirming(true))}
              >
                {decision === 'escalate' ? 'Escalate' : 'Record decision'}
              </Button>
            </Group>
          </>
        )}
      </Stack>

      <Modal
        opened={confirming}
        onClose={() => setConfirming(false)}
        title={`Close case ${c.id}?`}
        centered
      >
        <Stack>
          <Text size="sm">
            Record <b>{decision ? DECISION_LABEL[decision].toLowerCase() : ''}</b> and close the
            case. A closed case cannot be changed.
          </Text>
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setConfirming(false)}>
              Cancel
            </Button>
            <Button
              color={decision === 'confirmed_fraud' ? 'red' : 'teal'}
              loading={decide.isPending}
              onClick={submit}
            >
              Close case
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Paper>
  )
}
