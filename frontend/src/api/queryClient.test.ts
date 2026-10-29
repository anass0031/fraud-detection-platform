import { describe, expect, it } from 'vitest'

import { ApiError } from './client'
import { createQueryClient } from './queryClient'

describe('createQueryClient retry rule', () => {
  const retry = createQueryClient().getDefaultOptions().queries!.retry as (
    count: number,
    error: unknown,
  ) => boolean

  it('never retries client errors (they will not fix themselves)', () => {
    for (const status of [400, 401, 403, 404, 409, 422]) {
      expect(retry(0, new ApiError(status, 'x'))).toBe(false)
    }
  })

  it('retries server and network errors twice', () => {
    expect(retry(0, new ApiError(503, 'x'))).toBe(true)
    expect(retry(1, new TypeError('offline'))).toBe(true)
    expect(retry(2, new ApiError(500, 'x'))).toBe(false)
  })
})
