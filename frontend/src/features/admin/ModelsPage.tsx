import {
  Alert,
  Badge,
  Button,
  Center,
  Code,
  CopyButton,
  Group,
  List,
  Loader,
  Modal,
  Paper,
  SegmentedControl,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core'
import { useState } from 'react'

import { errorMessage } from '../../api/client'
import { localTime } from '../../utils/format'
import {
  type Comparison,
  type Scope,
  comparisonRows,
  confidenceText,
  readComparison,
} from './comparison'
import { type ModelVersion, useActivateModel, useArchiveModel, useModels } from './modelsApi'

const STATUS_COLOR = { active: 'green', candidate: 'blue', archived: 'gray' } as const

const RETRAIN_STEPS = [
  'docker compose exec -T backend python -m app.scripts.export_training_data --from 2018-04-28 --to 2018-05-26 > data/processed/replayed_labelled.parquet',
  'ml/.venv/Scripts/python ml/scripts/retrain.py',
  'docker compose exec backend python -m app.scripts.register_model <new version> --anomaly iforest-2026.10.27-1',
]

const windowText = (w: unknown) => (Array.isArray(w) && w.length === 2 ? `${w[0]} → ${w[1]}` : '—')

function ComparisonCard({ model, comparison }: { model: ModelVersion; comparison: Comparison }) {
  const [scope, setScope] = useState<Scope>('overall')
  const rows = comparisonRows(comparison, scope)
  const confidence = confidenceText(comparison)
  const side = comparison.champion[scope]
  return (
    <Paper withBorder p="md" aria-label={`Comparison for ${model.version}`}>
      <Stack gap="sm">
        <Group justify="space-between" align="flex-start">
          <div>
            <Title order={4}>
              {comparison.champion_version} vs {model.version}
            </Title>
            <Text size="xs" c="dimmed">
              Same transactions, never used for training: {comparison.window[0]} →{' '}
              {comparison.window[1]} (end excluded) · {side.n.toLocaleString('en-US')} transactions,{' '}
              {side.n_fraud.toLocaleString('en-US')} frauds
            </Text>
          </div>
          <SegmentedControl
            size="xs"
            value={scope}
            onChange={(v) => setScope(v as Scope)}
            data={[
              { value: 'overall', label: 'All customers' },
              { value: 'excl_fraud_customers', label: 'Excluding past-fraud customers' },
            ]}
          />
        </Group>
        <Table withTableBorder={false} verticalSpacing={6} aria-label="Champion vs challenger">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Metric</Table.Th>
              <Table.Th ta="right">Current ({comparison.champion_version})</Table.Th>
              <Table.Th ta="right">Candidate ({model.version})</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((r) => (
              <Table.Tr key={r.key}>
                <Table.Td>{r.label}</Table.Td>
                <Table.Td ta="right">{r.champion}</Table.Td>
                <Table.Td
                  ta="right"
                  fw={700}
                  c={r.better === null ? undefined : r.better ? 'teal.7' : 'red.7'}
                  data-better={r.better === null ? 'same' : r.better ? 'yes' : 'no'}
                >
                  {r.challenger}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
        <Text size="xs" c="dimmed">
          Alert thresholds (expected loss): current ${comparison.champion_threshold.toFixed(2)} ·
          candidate ${comparison.challenger_threshold.toFixed(2)}. Green = the candidate is better.
        </Text>
        {confidence && <Text size="sm">{confidence}</Text>}
      </Stack>
    </Paper>
  )
}

function ActivateDialog({
  model,
  current,
  onClose,
}: {
  model: ModelVersion | null
  current: ModelVersion | undefined
  onClose: () => void
}) {
  const activate = useActivateModel()
  return (
    <Modal
      opened={model !== null}
      onClose={onClose}
      title={`Activate ${model?.version ?? ''}?`}
      centered
    >
      {model && (
        <Stack>
          <List size="sm" spacing={4}>
            <List.Item>New transactions are scored with it immediately (no restart).</List.Item>
            {current && (
              <List.Item>
                {current.version} is archived (it can be activated again later).
              </List.Item>
            )}
            <List.Item>
              A custom alert threshold, if one is set, is reset to this model’s own threshold ($
              {model.threshold.toFixed(2)}).
            </List.Item>
            <List.Item>
              Monitoring follows the active model: its dashboard is empty until new days are
              replayed.
            </List.Item>
            <List.Item>The change is recorded in the audit trail.</List.Item>
          </List>
          <Group justify="flex-end">
            <Button variant="default" onClick={onClose}>
              Cancel
            </Button>
            <Button
              loading={activate.isPending}
              onClick={() => activate.mutate(model.id, { onSettled: onClose })}
            >
              Activate
            </Button>
          </Group>
        </Stack>
      )}
    </Modal>
  )
}

export function ModelsPage() {
  const models = useModels()
  const archive = useArchiveModel()
  const [toActivate, setToActivate] = useState<ModelVersion | null>(null)

  if (models.isPending) {
    return (
      <Center py="xl">
        <Loader aria-label="Loading models" />
      </Center>
    )
  }
  if (models.isError) {
    return (
      <Alert color="red" title="Could not load the models">
        <Stack gap="xs" align="flex-start">
          <Text size="sm">{errorMessage(models.error)}</Text>
          <Button size="xs" variant="light" onClick={() => void models.refetch()}>
            Retry
          </Button>
        </Stack>
      </Alert>
    )
  }

  const list = models.data
  const active = list.find((m) => m.status === 'active')
  const withComparison = list
    .filter((m) => m.status === 'candidate')
    .map((m) => ({ model: m, comparison: readComparison(m.metrics) }))
    .filter((x): x is { model: ModelVersion; comparison: Comparison } => x.comparison !== null)

  return (
    <Stack gap="md">
      <div>
        <Title order={2}>Models</Title>
        <Text c="dimmed" size="sm">
          Candidates come from the offline retraining script; compare them with the current model on
          recent data, then activate one (champion / challenger).
        </Text>
      </div>

      {withComparison.map(({ model, comparison }) => (
        <ComparisonCard key={model.id} model={model} comparison={comparison} />
      ))}

      <Paper withBorder>
        <Table.ScrollContainer minWidth={900}>
          <Table verticalSpacing="xs" aria-label="Model versions">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Version</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>Trained on</Table.Th>
                <Table.Th>Calibrated on</Table.Th>
                <Table.Th ta="right">Threshold</Table.Th>
                <Table.Th>Created</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {list.map((m) => {
                const w = (m.training_data_window ?? {}) as Record<string, unknown>
                return (
                  <Table.Tr key={m.id} aria-label={m.version}>
                    <Table.Td fw={600}>{m.version}</Table.Td>
                    <Table.Td>
                      <Group gap={6}>
                        <Badge color={STATUS_COLOR[m.status]} variant="light">
                          {m.status}
                        </Badge>
                        {m.serving && (
                          <Badge color="green" variant="dot">
                            serving
                          </Badge>
                        )}
                      </Group>
                    </Table.Td>
                    <Table.Td>{windowText(w.train)}</Table.Td>
                    <Table.Td>{windowText(w.calibration)}</Table.Td>
                    <Table.Td ta="right">${m.threshold.toFixed(2)}</Table.Td>
                    <Table.Td>{localTime(m.created_at)}</Table.Td>
                    <Table.Td>
                      <Group gap={6} justify="flex-end" wrap="nowrap">
                        {m.status !== 'active' && (
                          <Button size="compact-xs" onClick={() => setToActivate(m)}>
                            Activate
                          </Button>
                        )}
                        {m.status === 'candidate' && (
                          <Button
                            size="compact-xs"
                            variant="default"
                            loading={archive.isPending && archive.variables === m.id}
                            onClick={() => archive.mutate(m.id)}
                          >
                            Archive
                          </Button>
                        )}
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                )
              })}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      </Paper>

      <Paper withBorder p="md">
        <Title order={5}>Create a candidate (offline, from the repository root)</Title>
        <Text size="xs" c="dimmed" mb="xs">
          Exports the labels of replayed days, retrains, compares with the current model on the most
          recent week and registers the result as a candidate.
        </Text>
        <Stack gap={6}>
          {RETRAIN_STEPS.map((cmd, i) => (
            <Group key={cmd} gap="xs" wrap="nowrap" align="flex-start">
              <Text size="sm" w={16}>
                {i + 1}.
              </Text>
              <Code block style={{ flex: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {cmd}
              </Code>
              <CopyButton value={cmd}>
                {({ copied, copy }) => (
                  <Button size="compact-xs" variant="subtle" onClick={copy}>
                    {copied ? 'Copied' : 'Copy'}
                  </Button>
                )}
              </CopyButton>
            </Group>
          ))}
        </Stack>
      </Paper>

      <ActivateDialog model={toActivate} current={active} onClose={() => setToActivate(null)} />
    </Stack>
  )
}
