import { Alert, Group, Paper, RingProgress, Stack, Text, Title, Tooltip } from '@mantine/core'

import { localTime, money, percent } from '../../utils/format'
import { RiskBadge } from '../queue/badges'
import type { CaseDetail } from './api'

const RING_COLOR = { low: 'gray', medium: 'yellow', high: 'orange', critical: 'red' } as const

/** The model's verdict: calibrated probability, risk level, expected loss, unusual flag. */
export function ScoreCard({ c }: { c: CaseDetail }) {
  const p = c.prediction
  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <Title order={4}>Fraud score</Title>
        <Group wrap="nowrap" gap="lg">
          <RingProgress
            size={110}
            thickness={10}
            roundCaps
            sections={[{ value: Math.round(p.fraud_score * 100), color: RING_COLOR[p.risk_level] }]}
            label={
              <Text ta="center" fw={700} fz="xl" aria-label="Fraud probability">
                {percent(p.fraud_score)}
              </Text>
            }
          />
          <Stack gap={6}>
            <Group gap="xs">
              <Text size="sm" c="dimmed">
                Risk level
              </Text>
              <RiskBadge level={p.risk_level} />
            </Group>
            <Tooltip
              label="Fraud probability × amount. Cases are alerted when it reaches the threshold."
              withArrow
              multiline
              w={260}
            >
              <Group gap="xs">
                <Text size="sm" c="dimmed">
                  Expected loss
                </Text>
                <Text fw={700}>{money(p.expected_loss)}</Text>
              </Group>
            </Tooltip>
            <Text size="xs" c="dimmed">
              Model {p.model_version} · scored {localTime(p.scored_at)}
            </Text>
          </Stack>
        </Group>
        {p.is_unusual && (
          <Alert color="teal" variant="light" title="Unusual pattern">
            The fraud model did not alert on this transaction, but the anomaly detector found it
            very unusual compared with normal traffic. Review it with that in mind.
          </Alert>
        )}
      </Stack>
    </Paper>
  )
}
