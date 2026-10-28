import { Badge, Paper, Stack, Table, Text, Title } from '@mantine/core'

import { datasetTime, money } from '../../utils/format'
import type { CaseDetail } from './api'

/**
 * The customer's earlier transactions (same card/customer key, strictly before this one).
 * Only a "had a case" flag is shown, never earlier labels or decisions.
 */
export function HistoryCard({ c }: { c: CaseDetail }) {
  const rows = c.card_history
  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <div>
          <Title order={4}>Earlier transactions</Title>
          <Text size="xs" c="dimmed">
            Same customer, most recent first (up to 10)
          </Text>
        </div>
        {rows.length === 0 ? (
          <Text size="sm" c="dimmed">
            No earlier transactions for this customer: it is new, or could not be identified.
          </Text>
        ) : (
          <Table verticalSpacing={4} fz="sm">
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Time</Table.Th>
                <Table.Th ta="right">Amount</Table.Th>
                <Table.Th>Product</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {rows.map((h) => (
                <Table.Tr key={h.transaction_id}>
                  <Table.Td>{datasetTime(h.transaction_time)}</Table.Td>
                  <Table.Td ta="right">{money(h.amount)}</Table.Td>
                  <Table.Td>{h.product_code ?? '—'}</Table.Td>
                  <Table.Td>
                    {h.had_case && (
                      <Badge size="xs" variant="light" color="gray">
                        Had a case
                      </Badge>
                    )}
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        )}
      </Stack>
    </Paper>
  )
}
