import { Group, Table, Text, Tooltip, UnstyledButton } from '@mantine/core'
import {
  type ColumnDef,
  type SortingState,
  type Updater,
  rowPaginationFeature,
  rowSortingFeature,
  tableFeatures,
  useTable,
} from '@tanstack/react-table'
import { useNavigate } from 'react-router'

import { age, datasetTime, money, percent } from '../../utils/format'
import type { CaseRow } from './api'
import { ReasonBadge, RiskBadge, StatusBadge } from './badges'
import { CaseActions } from './CaseActions'
import type { Order, Sort } from './view'

// Sorting and paging are done by the API ("manual" mode); the table renders one page.
// A column is sortable only if it has an accessor (TanStack rule) and is in SORT_BY_COLUMN.
const features = tableFeatures({ rowSortingFeature, rowPaginationFeature })
type Features = typeof features

/** Sortable columns and the API sort key behind each one. */
const SORT_BY_COLUMN: Record<string, Sort> = {
  expected_loss: 'priority', // priority = tier, then expected loss
  amount: 'amount',
  transaction_time: 'transaction_time',
  age: 'created_at',
}
const COLUMN_BY_SORT = Object.fromEntries(
  Object.entries(SORT_BY_COLUMN).map(([column, sort]) => [sort, column]),
) as Record<Sort, string>

const baseColumns: ColumnDef<Features, CaseRow>[] = [
  {
    id: 'risk',
    header: 'Risk',
    cell: ({ row }) => <RiskBadge level={row.original.risk_level} />,
  },
  {
    id: 'score',
    header: 'Score',
    cell: ({ row }) => percent(row.original.fraud_score),
  },
  {
    id: 'expected_loss',
    accessorFn: (row) => row.expected_loss,
    header: () => (
      <Tooltip label="Fraud probability × amount: the queue's priority" withArrow>
        <span>Expected loss</span>
      </Tooltip>
    ),
    cell: ({ row }) => <Text fw={600}>{money(row.original.expected_loss)}</Text>,
  },
  {
    id: 'amount',
    accessorFn: (row) => row.amount,
    header: 'Amount',
    cell: ({ row }) => money(row.original.amount),
  },
  {
    id: 'transaction_time',
    accessorFn: (row) => row.transaction_time,
    header: 'Time',
    cell: ({ row }) => datasetTime(row.original.transaction_time),
  },
  {
    id: 'reason',
    header: 'Reason',
    cell: ({ row }) => <ReasonBadge reason={row.original.reason} />,
  },
  {
    id: 'status',
    header: 'Status',
    cell: ({ row }) => <StatusBadge status={row.original.status} />,
  },
  {
    id: 'assignee',
    header: 'Assignee',
    cell: ({ row }) =>
      row.original.assignee ? (
        row.original.assignee.full_name
      ) : (
        <Text c="dimmed" size="sm">
          Unassigned
        </Text>
      ),
  },
  {
    id: 'age',
    accessorFn: (row) => row.created_at,
    header: 'Age',
    cell: ({ row }) => (
      <Tooltip label={`Opened ${row.original.created_at}`} withArrow>
        <span>{age(row.original.created_at)}</span>
      </Tooltip>
    ),
  },
  {
    id: 'actions',
    header: '',
    cell: ({ row }) => <CaseActions row={row.original} />,
  },
]

const columns = baseColumns.map((col) => ({
  ...col,
  enableSorting: col.id !== undefined && col.id in SORT_BY_COLUMN,
}))

type Props = {
  rows: CaseRow[]
  total: number
  sort: Sort
  order: Order
  onSort: (sort: Sort, order: Order) => void
}

export function QueueTable({ rows, total, sort, order, onSort }: Props) {
  const navigate = useNavigate()
  const sorting: SortingState = [{ id: COLUMN_BY_SORT[sort], desc: order === 'desc' }]

  const onSortingChange = (updater: Updater<SortingState>) => {
    const next = typeof updater === 'function' ? updater(sorting) : updater
    const first = next[0]
    if (first && SORT_BY_COLUMN[first.id])
      onSort(SORT_BY_COLUMN[first.id], first.desc ? 'desc' : 'asc')
  }

  const table = useTable({
    features,
    columns,
    data: rows,
    rowCount: total,
    state: { sorting },
    onSortingChange,
    manualSorting: true,
    manualPagination: true,
    enableSortingRemoval: false, // a column toggles desc <-> asc, never "unsorted"
    sortDescFirst: true,
    enableMultiSort: false,
    getRowId: (row) => String(row.id),
  })

  return (
    <Table.ScrollContainer minWidth={980}>
      <Table highlightOnHover verticalSpacing="xs">
        <Table.Thead>
          {table.getHeaderGroups().map((group) => (
            <Table.Tr key={group.id}>
              {group.headers.map((header) => {
                const sorted = header.column.getIsSorted()
                const label = <table.FlexRender header={header} />
                return (
                  <Table.Th
                    key={header.id}
                    aria-sort={
                      sorted === 'asc' ? 'ascending' : sorted === 'desc' ? 'descending' : undefined
                    }
                  >
                    {header.column.getCanSort() ? (
                      <UnstyledButton
                        onClick={header.column.getToggleSortingHandler()}
                        fw={700}
                        fz="sm"
                      >
                        <Group gap={4} wrap="nowrap">
                          {label}
                          <Text span c={sorted ? undefined : 'dimmed'} fz="xs">
                            {sorted === 'asc' ? '▲' : sorted === 'desc' ? '▼' : '↕'}
                          </Text>
                        </Group>
                      </UnstyledButton>
                    ) : (
                      label
                    )}
                  </Table.Th>
                )
              })}
            </Table.Tr>
          ))}
        </Table.Thead>
        <Table.Tbody>
          {table.getRowModel().rows.map((row) => (
            <Table.Tr
              key={row.id}
              style={{ cursor: 'pointer' }}
              onClick={() => void navigate(`/cases/${row.original.id}`)}
              aria-label={`Case ${row.original.id}`}
            >
              {row.getAllCells().map((cell) => (
                <Table.Td key={cell.id}>
                  <Group gap={4} wrap="nowrap">
                    <table.FlexRender cell={cell} />
                  </Group>
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  )
}
