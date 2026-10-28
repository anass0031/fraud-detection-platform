import { Button, Group, Menu } from '@mantine/core'

import { useAuth } from '../auth/context'
import { useCan } from '../auth/useCan'
import { type CaseRow, useActiveUsers, useAssignCase, useClaimCase } from './api'

/** Claim (anyone allowed) and Assign (admins) for one case. Clicks never open the row. */
export function CaseActions({ row }: { row: CaseRow }) {
  const can = useCan()
  const { user } = useAuth()
  const claim = useClaimCase()
  const assign = useAssignCase()
  const canAssign = can.assign(row)
  const users = useActiveUsers(canAssign)
  // Escalated cases can only go to an admin.
  const candidates = (users.data ?? []).filter(
    (u) => (row.status !== 'escalated' || u.role === 'admin') && u.id !== row.assignee?.id,
  )

  return (
    <Group gap={6} wrap="nowrap" onClick={(e) => e.stopPropagation()}>
      {can.claim(row) && (
        <Button
          size="compact-xs"
          loading={claim.isPending}
          onClick={() => claim.mutate({ caseId: row.id })}
        >
          Claim
        </Button>
      )}
      {canAssign && (
        <Menu position="bottom-end" withinPortal>
          <Menu.Target>
            <Button size="compact-xs" variant="default" loading={assign.isPending}>
              Assign
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Label>Assign to</Menu.Label>
            {candidates.map((u) => (
              <Menu.Item key={u.id} onClick={() => assign.mutate({ caseId: row.id, userId: u.id })}>
                {u.full_name}
                {u.id === user?.id ? ' (me)' : ''}
              </Menu.Item>
            ))}
            {users.isPending && <Menu.Item disabled>Loading…</Menu.Item>}
            {users.isError && <Menu.Item disabled>Could not load users</Menu.Item>}
            {users.isSuccess && candidates.length === 0 && (
              <Menu.Item disabled>No one available</Menu.Item>
            )}
          </Menu.Dropdown>
        </Menu>
      )}
    </Group>
  )
}
