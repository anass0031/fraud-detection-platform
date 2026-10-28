import {
  Alert,
  Button,
  Center,
  Group,
  Loader,
  Pagination,
  Paper,
  Select,
  Stack,
  Tabs,
  Text,
  Title,
} from '@mantine/core'
import { useSearchParams } from 'react-router'

import { errorMessage } from '../../api/client'
import { useAuth } from '../auth/context'
import { useCases } from './api'
import { FilterBar } from './FilterBar'
import { QueueTable } from './QueueTable'
import {
  DEFAULT_VIEW,
  PAGE_SIZES,
  type QueueView,
  type Tab,
  hasFilters,
  parseView,
  updateView,
  viewToParams,
  viewToQuery,
} from './view'

const TAB_HELP: Record<Tab, string> = {
  open: 'Unclaimed cases: highest expected loss first, unusual-pattern cases after the model alerts.',
  mine: 'Cases you hold and still need to decide.',
  all: 'Every case, including closed ones.',
}

const EMPTY: Record<Tab, string> = {
  open: 'No open cases. The queue is empty.',
  mine: 'You hold no open cases. Claim one from the review queue.',
  all: 'No cases.',
}

export function QueuePage() {
  const { user } = useAuth()
  const [params, setParams] = useSearchParams()
  const view = parseView(params)
  const cases = useCases(viewToQuery(view, user!))

  const setView = (next: QueueView) => setParams(viewToParams(next))
  const change = (c: Partial<QueueView>) => setView(updateView(view, c))

  const total = cases.data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / view.size))
  const filtered = hasFilters(view)

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={2}>Review queue</Title>
          <Text c="dimmed" size="sm">
            {TAB_HELP[view.tab]}
          </Text>
        </div>
        {cases.data && (
          <Text size="sm" c="dimmed">
            {total.toLocaleString('en-US')} {total === 1 ? 'case' : 'cases'}
          </Text>
        )}
      </Group>

      <Tabs
        value={view.tab}
        onChange={(tab) =>
          setView({ ...DEFAULT_VIEW, tab: (tab as Tab) ?? 'open', size: view.size })
        }
      >
        <Tabs.List>
          <Tabs.Tab value="open">Open</Tabs.Tab>
          <Tabs.Tab value="mine">My cases</Tabs.Tab>
          <Tabs.Tab value="all">All</Tabs.Tab>
        </Tabs.List>
      </Tabs>

      <FilterBar
        view={view}
        onChange={change}
        onClear={() => setView({ ...DEFAULT_VIEW, tab: view.tab, size: view.size })}
      />

      {cases.isPending && (
        <Center py="xl">
          <Loader />
        </Center>
      )}
      {cases.isError && (
        <Alert color="red" title="Could not load the queue">
          <Stack gap="xs" align="flex-start">
            <Text size="sm">{errorMessage(cases.error)}</Text>
            <Button size="xs" variant="light" onClick={() => void cases.refetch()}>
              Retry
            </Button>
          </Stack>
        </Alert>
      )}
      {cases.data && cases.data.items.length === 0 && (
        <Paper withBorder p="xl">
          <Text c="dimmed" ta="center">
            {filtered ? 'No cases match these filters.' : EMPTY[view.tab]}
          </Text>
        </Paper>
      )}
      {cases.data && cases.data.items.length > 0 && (
        <Paper withBorder style={{ opacity: cases.isPlaceholderData ? 0.6 : 1 }}>
          <QueueTable
            rows={cases.data.items}
            total={total}
            sort={view.sort}
            order={view.order}
            onSort={(sort, order) => change({ sort, order })}
          />
        </Paper>
      )}

      {cases.data && total > 0 && (
        <Group justify="space-between">
          <Group gap="xs">
            <Text size="sm" c="dimmed">
              Rows per page
            </Text>
            <Select
              aria-label="Rows per page"
              data={PAGE_SIZES.map(String)}
              value={String(view.size)}
              onChange={(v) => change({ size: Number(v ?? DEFAULT_VIEW.size) })}
              w={80}
              allowDeselect={false}
            />
          </Group>
          <Pagination
            total={pages}
            value={Math.min(view.page, pages)}
            onChange={(page) => change({ page })}
            getControlProps={(control) => ({ 'aria-label': `${control} page` })}
          />
        </Group>
      )}
    </Stack>
  )
}
