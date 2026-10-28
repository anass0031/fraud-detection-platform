import { notifications } from '@mantine/notifications'
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { type Schemas, api, errorMessage, unwrap } from '../../api/client'
import type { paths } from '../../api/schema'

export type CaseRow = Schemas['CaseListItem']
export type CasesQuery = NonNullable<paths['/api/v1/cases']['get']['parameters']['query']>

/** One page of the review queue. Every request parameter is part of the cache key. */
export function useCases(query: CasesQuery) {
  return useQuery({
    queryKey: ['cases', query],
    queryFn: async () => unwrap(await api.GET('/api/v1/cases', { params: { query } })),
    // Keep the current page on screen while the next one loads (no flicker when paging).
    placeholderData: keepPreviousData,
  })
}

/** Active users (admin only: used by the Assign menu). */
export function useActiveUsers(enabled: boolean) {
  return useQuery({
    queryKey: ['users'],
    queryFn: async () => unwrap(await api.GET('/api/v1/users')),
    select: (users) => users.filter((u) => u.is_active),
    enabled,
    staleTime: 5 * 60_000,
  })
}

/**
 * After any case change: refresh the queue pages and that case. On error (e.g. someone else
 * claimed it first) the server's message is shown and the data refreshed, so the UI catches up.
 */
export function useCaseMutation<TVars extends { caseId: number }>(
  run: (vars: TVars) => Promise<CaseRow>,
  success: (row: CaseRow, vars: TVars) => string,
) {
  const queryClient = useQueryClient()
  const refresh = (caseId: number) =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: ['cases'] }),
      queryClient.invalidateQueries({ queryKey: ['case', caseId] }),
    ])
  return useMutation({
    mutationFn: run,
    onSuccess: (row, vars) => {
      notifications.show({ color: 'green', message: success(row, vars) })
      return refresh(vars.caseId)
    },
    onError: (error, vars) => {
      notifications.show({ color: 'red', title: 'Not saved', message: errorMessage(error) })
      return refresh(vars.caseId)
    },
  })
}

export function useClaimCase() {
  return useCaseMutation(
    async ({ caseId }: { caseId: number }) =>
      unwrap(
        await api.POST('/api/v1/cases/{case_id}/claim', {
          params: { path: { case_id: caseId } },
        }),
      ),
    (row) => `Case ${row.id} is now yours`,
  )
}

export function useAssignCase() {
  return useCaseMutation(
    async ({ caseId, userId }: { caseId: number; userId: number }) =>
      unwrap(
        await api.POST('/api/v1/cases/{case_id}/assign', {
          params: { path: { case_id: caseId } },
          body: { user_id: userId },
        }),
      ),
    (row) => `Case ${row.id} assigned to ${row.assignee?.full_name ?? 'nobody'}`,
  )
}
