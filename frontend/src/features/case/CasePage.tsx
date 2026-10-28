import {
  Alert,
  Anchor,
  Button,
  Center,
  Grid,
  Group,
  Loader,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import { Link, useParams } from 'react-router'

import { ApiError, errorMessage } from '../../api/client'
import { NotFoundPage } from '../../routes/ErrorPages'
import { CaseActions } from '../queue/CaseActions'
import { ReasonBadge, StatusBadge } from '../queue/badges'
import { useCase } from './api'
import { DecisionPanel } from './DecisionPanel'
import { ExplanationCard } from './ExplanationCard'
import { FactsCard } from './FactsCard'
import { HistoryCard } from './HistoryCard'
import { ScoreCard } from './ScoreCard'
import { DecisionsList, TimelineCard } from './TimelineCard'

export function CasePage() {
  const { caseId } = useParams()
  const id = Number(caseId)
  if (!Number.isInteger(id) || id <= 0) return <NotFoundPage />
  return <CaseView id={id} />
}

function CaseView({ id }: { id: number }) {
  const query = useCase(id)

  if (query.isPending) {
    return (
      <Center py="xl">
        <Loader aria-label="Loading case" />
      </Center>
    )
  }
  if (query.isError) {
    if (query.error instanceof ApiError && query.error.status === 404) return <NotFoundPage />
    return (
      <Alert color="red" title="Could not load the case">
        <Stack gap="xs" align="flex-start">
          <Text size="sm">{errorMessage(query.error)}</Text>
          <Button size="xs" variant="light" onClick={() => void query.refetch()}>
            Retry
          </Button>
        </Stack>
      </Alert>
    )
  }

  const c = query.data
  return (
    <Stack gap="md">
      <Anchor component={Link} to="/queue" size="sm">
        ← Review queue
      </Anchor>

      <Group justify="space-between" align="flex-start">
        <Stack gap={4}>
          <Group gap="sm">
            <Title order={2}>Case {c.id}</Title>
            <StatusBadge status={c.status} />
            <ReasonBadge reason={c.reason} />
          </Group>
          <Text size="sm" c="dimmed">
            {c.assignee ? `Assigned to ${c.assignee.full_name}` : 'Not assigned'}
            {c.final_decision &&
              ` · Decision: ${c.final_decision === 'confirmed_fraud' ? 'confirmed fraud' : 'false positive'}`}
          </Text>
        </Stack>
        <CaseActions row={c} />
      </Group>

      <Grid>
        <Grid.Col span={{ base: 12, md: 5 }}>
          <Stack gap="md">
            <ScoreCard c={c} />
            <FactsCard c={c} />
            <HistoryCard c={c} />
          </Stack>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 7 }}>
          <ExplanationCard c={c} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 7 }}>
          <Stack gap="md">
            <DecisionPanel c={c} />
            <DecisionsList c={c} />
          </Stack>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 5 }}>
          <TimelineCard c={c} />
        </Grid.Col>
      </Grid>
    </Stack>
  )
}
