import {
  Alert,
  Button,
  Center,
  Grid,
  Group,
  Image,
  Loader,
  NumberInput,
  Paper,
  Radio,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import { useState } from 'react'

import { errorMessage } from '../../api/client'
import { localTime, money } from '../../utils/format'
import {
  type Costs,
  type Thresholds,
  bandsError,
  costsError,
  thresholdError,
  useCosts,
  useSaveCosts,
  useSaveThresholds,
  useThresholds,
} from './settingsApi'

function Updated({
  at,
  by,
}: {
  at: string | null | undefined
  by: { full_name: string } | null | undefined
}) {
  return (
    <Text size="xs" c="dimmed">
      {at
        ? `Last changed ${localTime(at)}${by ? ` by ${by.full_name}` : ''}`
        : 'Default values (never changed)'}
    </Text>
  )
}

function ThresholdsForm({ data }: { data: Thresholds }) {
  const save = useSaveThresholds()
  const [custom, setCustom] = useState(data.alert_threshold !== null)
  const [value, setValue] = useState<number | string>(
    data.alert_threshold ?? data.model_threshold ?? '',
  )
  const [bands, setBands] = useState<(number | string)[]>(
    data.risk_bands.map((b) => +(b * 100).toFixed(2)),
  )

  const tError = thresholdError(custom, value)
  const bError = bandsError(bands)
  const modelThr = data.model_threshold == null ? '—' : money(data.model_threshold)

  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <div>
          <Title order={4}>Alert threshold</Title>
          <Text size="xs" c="dimmed">
            A transaction is alerted when its expected loss (fraud probability × amount) reaches the
            threshold. Changes apply to transactions scored afterwards.
          </Text>
        </div>
        <Text size="sm">
          In force now:{' '}
          <b>{data.effective_threshold == null ? '—' : money(data.effective_threshold)}</b> (model{' '}
          {data.model_version ?? '—'}, own threshold {modelThr})
        </Text>
        <Radio.Group
          value={custom ? 'custom' : 'model'}
          onChange={(v) => setCustom(v === 'custom')}
        >
          <Stack gap="xs">
            <Radio value="model" label={`Use the model’s own threshold (${modelThr})`} />
            <Radio value="custom" label="Custom threshold" />
          </Stack>
        </Radio.Group>
        {custom && (
          <NumberInput
            label="Custom threshold (expected loss, USD)"
            prefix="$"
            decimalScale={2}
            min={0}
            value={value}
            onChange={setValue}
            error={tError}
            w={240}
          />
        )}

        <Title order={5} mt="xs">
          Risk levels (calibrated probability)
        </Title>
        <Group gap="sm" align="flex-start">
          {['Low below', 'Medium below', 'High below'].map((label, i) => (
            <NumberInput
              key={label}
              label={label}
              suffix="%"
              decimalScale={2}
              min={0}
              max={100}
              value={bands[i]}
              onChange={(v) => setBands((b) => b.map((x, j) => (j === i ? v : x)))}
              w={130}
            />
          ))}
        </Group>
        <Text size="xs" c={bError ? 'red' : 'dimmed'}>
          {bError ?? 'Above the last bound: critical.'}
        </Text>

        <Group justify="space-between">
          <Updated at={data.updated_at} by={data.updated_by} />
          <Button
            disabled={Boolean(tError || bError)}
            loading={save.isPending}
            onClick={() =>
              save.mutate({
                alert_threshold: custom ? Number(value) : null,
                risk_bands: bands.map((b) => Number(b) / 100),
              })
            }
          >
            Save thresholds
          </Button>
        </Group>
      </Stack>
    </Paper>
  )
}

function CostsForm({ data }: { data: Costs }) {
  const save = useSaveCosts()
  const [review, setReview] = useState<number | string>(data.review_cost)
  const [friction, setFriction] = useState<number | string>(data.friction_cost)
  const [capacity, setCapacity] = useState<number | string>(data.capacity_per_day)
  const error = costsError({ review, friction, capacity })

  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <div>
          <Title order={4}>Business costs</Title>
          <Text size="xs" c="dimmed">
            Used by the monitoring (net savings, precision within capacity). A missed fraud costs
            its amount.
          </Text>
        </div>
        <NumberInput
          label="Review cost per alert"
          prefix="$"
          decimalScale={2}
          min={0}
          value={review}
          onChange={setReview}
          w={220}
        />
        <NumberInput
          label="Friction cost per false alarm"
          prefix="$"
          decimalScale={2}
          min={0}
          value={friction}
          onChange={setFriction}
          w={220}
        />
        <NumberInput
          label="Review capacity (cases per day)"
          min={1}
          allowDecimal={false}
          value={capacity}
          onChange={setCapacity}
          w={220}
        />
        {error && (
          <Text size="xs" c="red">
            {error}
          </Text>
        )}
        <Group justify="space-between">
          <Updated at={data.updated_at} by={data.updated_by} />
          <Button
            disabled={Boolean(error)}
            loading={save.isPending}
            onClick={() =>
              save.mutate({
                review_cost: Number(review),
                friction_cost: Number(friction),
                capacity_per_day: Number(capacity),
              })
            }
          >
            Save costs
          </Button>
        </Group>
      </Stack>
    </Paper>
  )
}

export function SettingsPage() {
  const thresholds = useThresholds()
  const costs = useCosts()

  if (thresholds.isPending || costs.isPending) {
    return (
      <Center py="xl">
        <Loader aria-label="Loading settings" />
      </Center>
    )
  }
  const failed = [thresholds, costs].find((q) => q.isError)
  if (failed || !thresholds.data || !costs.data) {
    return (
      <Alert color="red" title="Could not load the settings">
        <Stack gap="xs" align="flex-start">
          <Text size="sm">{errorMessage(failed?.error)}</Text>
          <Button
            size="xs"
            variant="light"
            onClick={() => [thresholds, costs].forEach((q) => void q.refetch())}
          >
            Retry
          </Button>
        </Stack>
      </Alert>
    )
  }

  return (
    <Stack gap="md">
      <div>
        <Title order={2}>Threshold and costs</Title>
        <Text c="dimmed" size="sm">
          Every change is recorded in the audit trail.
        </Text>
      </div>
      <Grid>
        <Grid.Col span={{ base: 12, lg: 6 }}>
          <Stack gap="md">
            {/* key: rebuild the form from the saved values after each save */}
            <ThresholdsForm key={thresholds.data.updated_at ?? 'default'} data={thresholds.data} />
            <CostsForm key={costs.data.updated_at ?? 'default'} data={costs.data} />
          </Stack>
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 6 }}>
          <Paper withBorder p="md">
            <Title order={4}>Why this threshold</Title>
            <Text size="xs" c="dimmed" mb="sm">
              From training (model lgbm-2026.10.27-1, validation period): total cost for each number
              of alerts per day. The production rule is the orange curve (expected loss); its
              threshold is where the curve meets the 150/day capacity line. Lower thresholds catch
              more fraud but need more analysts.
            </Text>
            <Image
              src="/figures/model_cost_policies.png"
              alt="Total cost versus mean alerts per day for the two alert policies"
              radius="sm"
            />
          </Paper>
        </Grid.Col>
      </Grid>
    </Stack>
  )
}
