import { Paper, SimpleGrid, Stack, Text, Title } from '@mantine/core'

import { datasetTime, money } from '../../utils/format'
import type { CaseDetail } from './api'

function Fact({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text size="sm" c={value ? undefined : 'dimmed'}>
        {value || '—'}
      </Text>
    </div>
  )
}

/** What happened: the transaction as received (no label, no ground truth). */
export function FactsCard({ c }: { c: CaseDetail }) {
  const t = c.transaction
  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <Title order={4}>Transaction</Title>
        <SimpleGrid cols={2} spacing="sm" verticalSpacing="xs">
          <Fact label="Amount" value={money(t.amount)} />
          <Fact label="Time" value={datasetTime(t.transaction_time)} />
          <Fact label="Transaction ID" value={String(t.transaction_id)} />
          <Fact label="Product" value={t.product_code} />
          <Fact label="Card network" value={t.card_network} />
          <Fact label="Card type" value={t.card_type} />
          <Fact label="Payer e-mail domain" value={t.payer_email_domain} />
          <Fact label="Device type" value={t.device_type} />
        </SimpleGrid>
      </Stack>
    </Paper>
  )
}
