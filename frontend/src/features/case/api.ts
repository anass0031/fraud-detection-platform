import { useQuery } from '@tanstack/react-query'

import { type Schemas, api, unwrap } from '../../api/client'
import { useCaseMutation } from '../queue/api'

export type CaseDetail = Schemas['CaseDetail']
export type TimelineEntry = Schemas['TimelineEntry']
export type DecisionKind = Schemas['DecisionRequest']['decision']

/** One case with its transaction, prediction, explanation, card history and decisions. */
export function useCase(caseId: number) {
  return useQuery({
    queryKey: ['case', caseId],
    queryFn: async () =>
      unwrap(await api.GET('/api/v1/cases/{case_id}', { params: { path: { case_id: caseId } } })),
  })
}

/** The case's audit trail, oldest first. Its key starts with ['case', id], so every case
 * change (claim, assign, decision) refreshes it too. */
export function useTimeline(caseId: number) {
  return useQuery({
    queryKey: ['case', caseId, 'timeline'],
    queryFn: async () =>
      unwrap(
        await api.GET('/api/v1/cases/{case_id}/timeline', {
          params: { path: { case_id: caseId } },
        }),
      ),
  })
}

export const DECISION_LABEL: Record<DecisionKind, string> = {
  confirmed_fraud: 'Confirmed fraud',
  false_positive: 'False positive',
  escalate: 'Escalated',
}

export function useDecide() {
  return useCaseMutation(
    async ({
      caseId,
      decision,
      comment,
    }: {
      caseId: number
      decision: DecisionKind
      comment: string | null
    }) =>
      unwrap(
        await api.POST('/api/v1/cases/{case_id}/decision', {
          params: { path: { case_id: caseId } },
          body: { decision, comment },
        }),
      ),
    (row, vars) =>
      vars.decision === 'escalate'
        ? `Case ${row.id} escalated to the admins`
        : `Case ${row.id} closed: ${DECISION_LABEL[vars.decision].toLowerCase()}`,
  )
}
