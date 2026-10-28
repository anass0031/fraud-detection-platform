import { BarChart, LineChart } from '@mantine/charts'
import { Badge, Group, Paper, SimpleGrid, Stack, Table, Text, Title, Tooltip } from '@mantine/core'

import type { Alerts, Drift, Feedback, Performance } from './api'
import {
  BAND_COLOR,
  type Kpi,
  alertSeries,
  heatmapRows,
  kpis,
  performanceSeries,
  psiBand,
} from './shape'

function KpiCard({ k }: { k: Kpi }) {
  const good = k.delta == null || k.delta === 0 ? null : k.delta > 0 === (k.better === 'up')
  return (
    <Paper withBorder p="md">
      <Tooltip label={k.help} withArrow multiline w={240}>
        <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
          {k.label}
        </Text>
      </Tooltip>
      <Text fz={26} fw={700} aria-label={`${k.label} last 7 days`}>
        {k.value}
      </Text>
      <Text size="xs" c={good === null ? 'dimmed' : good ? 'teal.7' : 'red.7'}>
        whole period: {k.previous}
      </Text>
    </Paper>
  )
}

export function KpiCards({ perf }: { perf: Performance }) {
  if (!perf.total || !perf.last_7_days) return null
  return (
    <SimpleGrid cols={{ base: 2, sm: 3, lg: 6 }} spacing="md">
      {kpis(perf.last_7_days, perf.total).map((k) => (
        <KpiCard key={k.key} k={k} />
      ))}
    </SimpleGrid>
  )
}

export function PerformanceCard({ perf }: { perf: Performance }) {
  return (
    <Paper withBorder p="md">
      <Title order={4}>Performance per day</Title>
      <Text size="xs" c="dimmed" mb="sm">
        On replayed days, using their revealed labels (%). Daily values are noisy: about 80 frauds a
        day.
      </Text>
      <LineChart
        h={260}
        data={performanceSeries(perf)}
        dataKey="day"
        series={[
          { name: 'Precision', color: 'indigo.6' },
          { name: 'Recall', color: 'teal.6' },
          { name: 'PR-AUC', color: 'grape.6' },
          { name: 'Fraud $ caught', color: 'orange.6' },
        ]}
        withLegend
        connectNulls
        curveType="monotone"
        valueFormatter={(v) => `${v.toFixed(1)}%`}
        yAxisProps={{ domain: [0, 100] }}
      />
    </Paper>
  )
}

export function AlertsCard({ alerts }: { alerts: Alerts }) {
  const open = Object.values(alerts.backlog).reduce((a, b) => a + b, 0)
  return (
    <Paper withBorder p="md">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={4}>Alert volume</Title>
          <Text size="xs" c="dimmed" mb="sm">
            Cases opened per transaction day; dashed line = review capacity (
            {alerts.capacity_per_day}/day)
          </Text>
        </div>
        <Stack gap={0} align="flex-end">
          <Text fw={700} aria-label="Open backlog">
            {open.toLocaleString('en-US')}
          </Text>
          <Text size="xs" c="dimmed">
            open cases ({alerts.backlog.new} new, {alerts.backlog.assigned} assigned,{' '}
            {alerts.backlog.escalated} escalated)
          </Text>
        </Stack>
      </Group>
      <BarChart
        h={220}
        data={alertSeries(alerts)}
        dataKey="day"
        type="stacked"
        series={[
          { name: 'Model alerts', color: 'indigo.6' },
          { name: 'Unusual pattern', color: 'teal.5' },
        ]}
        withLegend
        referenceLines={[
          { y: alerts.capacity_per_day, color: 'red.6', label: 'Capacity', strokeDasharray: '5 5' },
        ]}
      />
    </Paper>
  )
}

export function DriftCard({ drift }: { drift: Drift }) {
  const rows = heatmapRows(drift)
  return (
    <Paper withBorder p="md">
      <Title order={4}>Data drift (PSI against the training data)</Title>
      <Text size="xs" c="dimmed" mb="sm">
        Model score and the 15 most drifting features. PSI &lt; {drift.thresholds.moderate} stable ·
        up to {drift.thresholds.significant} moderate · above: significant. The “days since”
        features grow with the calendar, so they drift by nature; what matters most is whether the
        score drifts.
      </Text>
      <Table.ScrollContainer minWidth={420 + drift.days.length * 26}>
        <Table
          withRowBorders={false}
          verticalSpacing={2}
          horizontalSpacing={2}
          aria-label="Drift heatmap"
        >
          <Table.Thead>
            <Table.Tr>
              <Table.Th />
              {drift.days.map((d) => (
                <Table.Th key={d} fz={10} c="dimmed" ta="center" fw={400}>
                  {d.slice(8)}
                </Table.Th>
              ))}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((r) => (
              <Table.Tr key={r.name}>
                <Table.Td
                  fz="xs"
                  pr="sm"
                  style={{ whiteSpace: 'nowrap' }}
                  fw={r.name === '__score__' ? 700 : 400}
                >
                  {r.label}
                </Table.Td>
                {drift.days.map((d) => {
                  const psi = r.cells.get(d)
                  const band = psi == null ? null : psiBand(psi, drift.thresholds)
                  return (
                    <Table.Td key={d} p={1}>
                      <Tooltip
                        label={
                          psi == null ? 'no data' : `${r.label} · ${d} · PSI ${psi.toFixed(3)}`
                        }
                      >
                        <div
                          data-band={band ?? 'none'}
                          style={{
                            width: 22,
                            height: 18,
                            borderRadius: 3,
                            background: band
                              ? BAND_COLOR[band]
                              : 'var(--mantine-color-default-border)',
                          }}
                        />
                      </Tooltip>
                    </Table.Td>
                  )
                })}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Group gap="md" mt="xs">
        {(['stable', 'moderate', 'significant'] as const).map((b) => (
          <Group key={b} gap={4}>
            <div style={{ width: 12, height: 12, borderRadius: 2, background: BAND_COLOR[b] }} />
            <Text size="xs">{b}</Text>
          </Group>
        ))}
      </Group>
    </Paper>
  )
}

export function FeedbackCard({ feedback }: { feedback: Feedback }) {
  const agreement =
    feedback.checked_against_labels > 0
      ? `${feedback.matching_labels} of ${feedback.checked_against_labels}`
      : '—'
  return (
    <Paper withBorder p="md">
      <Title order={4}>Analyst feedback</Title>
      <Text size="xs" c="dimmed" mb="sm">
        Decisions recorded in the review queue
      </Text>
      <Group gap="xs" mb="sm">
        <Badge color="red" variant="light">
          {feedback.decisions.confirmed_fraud} confirmed fraud
        </Badge>
        <Badge color="teal" variant="light">
          {feedback.decisions.false_positive} false positive
        </Badge>
        <Badge color="grape" variant="light">
          {feedback.decisions.escalate} escalated
        </Badge>
      </Group>
      <Stack gap={4}>
        <Text size="sm">
          Closed cases: <b>{feedback.cases_closed}</b>
        </Text>
        <Text size="sm">
          Confirmed as fraud:{' '}
          <b>
            {feedback.confirmed_share == null
              ? '—'
              : `${(feedback.confirmed_share * 100).toFixed(0)}%`}
          </b>
        </Text>
        <Text size="sm">
          Decisions matching the revealed label: <b>{agreement}</b>
        </Text>
        <Text size="xs" c="dimmed">
          Mean time to decision:{' '}
          {feedback.mean_hours_to_decision == null
            ? '—'
            : `${feedback.mean_hours_to_decision.toFixed(1)} h`}{' '}
          (not meaningful for replayed days: their cases are opened in bulk)
        </Text>
      </Stack>
    </Paper>
  )
}
