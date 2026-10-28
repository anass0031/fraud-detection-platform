import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { type Schemas, api, errorMessage, unwrap } from '../../api/client'

export type Thresholds = Schemas['ThresholdsOut']
export type ThresholdsIn = Schemas['ThresholdsIn']
export type Costs = Schemas['CostsOut']
export type CostsIn = Schemas['CostsIn']

export function useThresholds() {
  return useQuery({
    queryKey: ['settings', 'thresholds'],
    queryFn: async () => unwrap(await api.GET('/api/v1/settings/thresholds')),
  })
}

export function useCosts() {
  return useQuery({
    queryKey: ['settings', 'costs'],
    queryFn: async () => unwrap(await api.GET('/api/v1/settings/costs')),
  })
}

function useSave<T>(save: (body: T) => Promise<unknown>, what: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: save,
    onSuccess: () => {
      notifications.show({ color: 'green', message: `${what} saved (recorded in the audit trail)` })
      return queryClient.invalidateQueries({ queryKey: ['settings'] })
    },
    onError: (error) =>
      notifications.show({
        color: 'red',
        title: `${what} not saved`,
        message: errorMessage(error),
      }),
  })
}

export function useSaveThresholds() {
  return useSave(
    async (body: ThresholdsIn) => unwrap(await api.PUT('/api/v1/settings/thresholds', { body })),
    'Threshold settings',
  )
}

export function useSaveCosts() {
  return useSave(
    async (body: CostsIn) => unwrap(await api.PUT('/api/v1/settings/costs', { body })),
    'Cost settings',
  )
}

// ---------------------------------------------------------------- validation (same rules as the API)

/** Problems with the risk-band bounds (in %), or null when they are valid. */
export function bandsError(bandsPct: (number | string)[]): string | null {
  const v = bandsPct.map(Number)
  if (v.some((x) => !Number.isFinite(x) || x <= 0 || x >= 100))
    return 'Each bound must be between 0% and 100%'
  if (!(v[0] < v[1] && v[1] < v[2])) return 'Bounds must increase: low < medium < high'
  return null
}

export function thresholdError(custom: boolean, value: number | string): string | null {
  if (!custom) return null
  const v = Number(value)
  if (value === '' || !Number.isFinite(v) || v <= 0) return 'Enter an amount above $0'
  if (v > 100_000) return 'At most $100,000'
  return null
}

export function costsError(c: {
  review: number | string
  friction: number | string
  capacity: number | string
}) {
  const review = Number(c.review)
  const friction = Number(c.friction)
  const capacity = Number(c.capacity)
  if (c.review === '' || !Number.isFinite(review) || review < 0 || review > 10_000)
    return 'Review cost must be between $0 and $10,000'
  if (c.friction === '' || !Number.isFinite(friction) || friction < 0 || friction > 10_000)
    return 'Friction cost must be between $0 and $10,000'
  if (c.capacity === '' || !Number.isInteger(capacity) || capacity < 1 || capacity > 10_000)
    return 'Capacity must be a whole number between 1 and 10,000'
  return null
}
