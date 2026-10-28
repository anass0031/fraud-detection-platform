import { Alert, Button, Center, Grid, Loader, Paper, Stack, Text, Title } from '@mantine/core'

import { errorMessage } from '../../api/client'
import { useAlertVolume, useDrift, useFeedback, usePerformance } from './api'
import { AlertsCard, DriftCard, FeedbackCard, KpiCards, PerformanceCard } from './MonitoringCards'

export function MonitoringPage() {
  const perf = usePerformance()
  const drift = useDrift()
  const alerts = useAlertVolume()
  const feedback = useFeedback()
  const queries = [perf, drift, alerts, feedback]

  const header = (
    <div>
      <Title order={2}>Monitoring</Title>
      <Text c="dimmed" size="sm">
        {perf.data?.total
          ? `Model ${perf.data.model_version} on ${perf.data.total.days} replayed days ` +
            `(${perf.data.total.first_day} → ${perf.data.total.last_day}). Cards: last 7 days.`
          : 'Model performance, data drift and alert volume on replayed days.'}
      </Text>
    </div>
  )

  if (queries.some((q) => q.isPending)) {
    return (
      <Stack>
        {header}
        <Center py="xl">
          <Loader aria-label="Loading monitoring" />
        </Center>
      </Stack>
    )
  }
  const failed = queries.find((q) => q.isError)
  if (failed) {
    return (
      <Stack>
        {header}
        <Alert color="red" title="Could not load monitoring">
          <Stack gap="xs" align="flex-start">
            <Text size="sm">{errorMessage(failed.error)}</Text>
            <Button
              size="xs"
              variant="light"
              onClick={() => queries.forEach((q) => void q.refetch())}
            >
              Retry
            </Button>
          </Stack>
        </Alert>
      </Stack>
    )
  }
  if (!perf.data || !drift.data || !alerts.data || !feedback.data) return null

  if (perf.data.days.length === 0) {
    return (
      <Stack>
        {header}
        <Paper withBorder p="xl">
          <Text ta="center" c="dimmed">
            No replayed day yet. Monitoring appears after the replay script has loaded, scored and
            revealed the labels of at least one day.
          </Text>
        </Paper>
      </Stack>
    )
  }

  return (
    <Stack gap="md">
      {header}
      <KpiCards perf={perf.data} />
      <Grid>
        <Grid.Col span={{ base: 12, lg: 7 }}>
          <PerformanceCard perf={perf.data} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 5 }}>
          <AlertsCard alerts={alerts.data} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 9 }}>
          <DriftCard drift={drift.data} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 3 }}>
          <FeedbackCard feedback={feedback.data} />
        </Grid.Col>
      </Grid>
    </Stack>
  )
}
