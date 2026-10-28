import { useMemo } from 'react'

import type { Schemas } from '../../api/client'
import { useAuth } from './context'
import { canAssign, canClaim, canDecide, canEscalate } from './permissions'

type CaseLike = Pick<Schemas['CaseListItem'], 'status' | 'assignee'>
const never = () => false

/** What the logged-in user may do with a case (UI only; the API re-checks everything). */
export function useCan() {
  const { user } = useAuth()
  return useMemo(
    () =>
      user === null
        ? { claim: never, assign: never, decide: never, escalate: never }
        : {
            claim: (c: CaseLike) => canClaim(user, c),
            assign: (c: CaseLike) => canAssign(user, c),
            decide: (c: CaseLike) => canDecide(user, c),
            escalate: (c: CaseLike) => canEscalate(user, c),
          },
    [user],
  )
}
