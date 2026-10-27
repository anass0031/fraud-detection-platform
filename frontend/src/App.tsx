import { Badge, Button, Card, Container, Group, Loader, Stack, Text, Title } from '@mantine/core'
import { useCallback, useEffect, useState } from 'react'

type Health = {
  status: 'ok' | 'degraded'
  database: 'ok' | 'error'
  version: string
}

type HealthState =
  { kind: 'loading' } | { kind: 'ok'; health: Health } | { kind: 'error'; message: string }

async function fetchHealth(): Promise<Health> {
  const resp = await fetch('/api/v1/health')
  // 503 still carries a JSON body describing what is down.
  if (resp.status !== 200 && resp.status !== 503) {
    throw new Error(`Unexpected HTTP ${resp.status}`)
  }
  return (await resp.json()) as Health
}

function StatusBadge({ label, ok }: { label: string; ok: boolean }) {
  return (
    <Badge size="lg" color={ok ? 'green' : 'red'} variant="light">
      {label} {ok ? 'OK' : 'DOWN'}
    </Badge>
  )
}

export default function App() {
  const [state, setState] = useState<HealthState>({ kind: 'loading' })

  // State is only updated when the request finishes (initial state is already "loading").
  const fetchAndStore = useCallback(
    () =>
      fetchHealth().then(
        (health) => setState({ kind: 'ok', health }),
        (err: unknown) =>
          setState({ kind: 'error', message: err instanceof Error ? err.message : String(err) }),
      ),
    [],
  )

  useEffect(() => {
    void fetchAndStore()
  }, [fetchAndStore])

  const refresh = () => {
    setState({ kind: 'loading' })
    void fetchAndStore()
  }

  return (
    <Container size="sm" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Fraud Detection &amp; Explainable AI</Title>
          <Text c="dimmed">Platform skeleton: system status</Text>
        </div>

        <Card withBorder padding="lg">
          <Stack gap="md">
            <Title order={3}>System status</Title>
            {state.kind === 'loading' && <Loader size="sm" />}
            {state.kind === 'error' && (
              <Group>
                <StatusBadge label="Backend" ok={false} />
                <Text size="sm" c="red">
                  {state.message}
                </Text>
              </Group>
            )}
            {state.kind === 'ok' && (
              <Group>
                <StatusBadge label="Backend" ok />
                <StatusBadge label="Database" ok={state.health.database === 'ok'} />
                <Text size="sm" c="dimmed">
                  API v{state.health.version}
                </Text>
              </Group>
            )}
            <Group>
              <Button variant="default" onClick={refresh}>
                Refresh
              </Button>
            </Group>
          </Stack>
        </Card>
      </Stack>
    </Container>
  )
}
