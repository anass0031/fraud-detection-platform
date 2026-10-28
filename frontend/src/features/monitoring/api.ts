import { useQuery } from '@tanstack/react-query'

import { type Schemas, api, unwrap } from '../../api/client'

export type Performance = Schemas['PerformanceResponse']
export type PerformanceTotals = Schemas['PerformanceTotals']
export type Drift = Schemas['DriftResponse']
export type Alerts = Schemas['AlertsResponse']
export type Feedback = Schemas['FeedbackResponse']

// Monitoring changes only when days are replayed: no need to refetch often.
const STALE = 60_000

export function usePerformance() {
  return useQuery({
    queryKey: ['monitoring', 'performance'],
    queryFn: async () => unwrap(await api.GET('/api/v1/monitoring/performance')),
    staleTime: STALE,
  })
}

export function useDrift() {
  return useQuery({
    queryKey: ['monitoring', 'drift'],
    queryFn: async () => unwrap(await api.GET('/api/v1/monitoring/drift')),
    staleTime: STALE,
  })
}

export function useAlertVolume() {
  return useQuery({
    queryKey: ['monitoring', 'alerts'],
    queryFn: async () => unwrap(await api.GET('/api/v1/monitoring/alerts')),
    staleTime: STALE,
  })
}

export function useFeedback() {
  return useQuery({
    queryKey: ['monitoring', 'feedback'],
    queryFn: async () => unwrap(await api.GET('/api/v1/monitoring/feedback')),
    staleTime: STALE,
  })
}
