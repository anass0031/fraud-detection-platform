import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ApiError,
  authFetch,
  errorMessage,
  getAccessToken,
  onSessionExpired,
  refreshSession,
  setAccessToken,
  unwrap,
} from './client'

const USER = { id: 1, email: 'a@x.io', full_name: 'A', role: 'analyst', is_active: true }

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function session(token: string) {
  return json(200, { access_token: token, token_type: 'bearer', expires_in: 900, user: USER })
}

function path(input: RequestInfo | URL): string {
  const url = input instanceof Request ? input.url : String(input)
  return new URL(url, window.location.origin).pathname
}

function authHeader(input: RequestInfo | URL): string | null {
  return input instanceof Request ? input.headers.get('Authorization') : null
}

const fetchMock = vi.fn<typeof fetch>()

beforeEach(() => {
  setAccessToken(null)
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => vi.unstubAllGlobals())

describe('refreshSession', () => {
  it('shares one request between concurrent callers', async () => {
    fetchMock.mockResolvedValue(session('new-token'))
    const [a, b, c] = await Promise.all([refreshSession(), refreshSession(), refreshSession()])
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(a?.access_token).toBe('new-token')
    expect(b).toBe(a)
    expect(c).toBe(a)
    expect(getAccessToken()).toBe('new-token')
  })

  it('starts a new request once the previous one has finished', async () => {
    fetchMock.mockImplementation(async () => session('t'))
    await refreshSession()
    await refreshSession()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('resolves to null and clears the token when there is no session', async () => {
    setAccessToken('old')
    fetchMock.mockResolvedValue(json(401, { detail: 'Invalid refresh token' }))
    expect(await refreshSession()).toBeNull()
    expect(getAccessToken()).toBeNull()
  })

  it('resolves to null on a network error', async () => {
    fetchMock.mockRejectedValue(new TypeError('offline'))
    expect(await refreshSession()).toBeNull()
  })
})

describe('authFetch', () => {
  it('sends the access token', async () => {
    setAccessToken('abc')
    fetchMock.mockResolvedValue(json(200, {}))
    await authFetch(new Request('http://localhost/api/v1/cases'))
    expect(authHeader(fetchMock.mock.calls[0][0])).toBe('Bearer abc')
  })

  it('refreshes once on 401 and retries with the new token and the same body', async () => {
    setAccessToken('expired')
    fetchMock.mockImplementation(async (input) => {
      if (path(input) === '/api/v1/auth/refresh') return session('fresh')
      return authHeader(input) === 'Bearer fresh' ? json(200, { ok: true }) : json(401, {})
    })
    const resp = await authFetch(
      new Request('http://localhost/api/v1/cases/1/decision', {
        method: 'POST',
        body: JSON.stringify({ decision: 'false_positive' }),
      }),
    )
    expect(resp.status).toBe(200)
    const calls = fetchMock.mock.calls.map(([input]) => path(input))
    expect(calls).toEqual([
      '/api/v1/cases/1/decision',
      '/api/v1/auth/refresh',
      '/api/v1/cases/1/decision',
    ])
    const retried = fetchMock.mock.calls[2][0] as Request
    expect(await retried.text()).toBe('{"decision":"false_positive"}')
  })

  it('uses a single refresh when several requests get 401 together', async () => {
    setAccessToken('expired')
    fetchMock.mockImplementation(async (input) => {
      if (path(input) === '/api/v1/auth/refresh') return session('fresh')
      return authHeader(input) === 'Bearer fresh' ? json(200, {}) : json(401, {})
    })
    await Promise.all(
      [1, 2, 3].map((id) => authFetch(new Request(`http://localhost/api/v1/cases/${id}`))),
    )
    const refreshes = fetchMock.mock.calls.filter(([i]) => path(i) === '/api/v1/auth/refresh')
    expect(refreshes).toHaveLength(1)
  })

  it('signals an expired session when the refresh fails', async () => {
    const expired = vi.fn()
    const unsubscribe = onSessionExpired(expired)
    fetchMock.mockResolvedValue(json(401, {}))
    const resp = await authFetch(new Request('http://localhost/api/v1/cases'))
    unsubscribe()
    expect(resp.status).toBe(401)
    expect(expired).toHaveBeenCalledTimes(1)
  })

  it('never refreshes for the auth endpoints themselves (wrong password stays a 401)', async () => {
    fetchMock.mockResolvedValue(json(401, { detail: 'Incorrect e-mail or password' }))
    const resp = await authFetch(
      new Request('http://localhost/api/v1/auth/login', { method: 'POST' }),
    )
    expect(resp.status).toBe(401)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('errors', () => {
  it('reads FastAPI error details', () => {
    expect(errorMessage({ detail: 'Case is closed' })).toBe('Case is closed')
    expect(errorMessage({ detail: [{ msg: 'field required' }] })).toBe('field required')
    expect(errorMessage(undefined, 'fallback')).toBe('fallback')
  })

  it('unwrap returns data or throws ApiError with the status', () => {
    const ok = new Response(null, { status: 200 })
    expect(unwrap({ data: 1, response: ok })).toBe(1)
    const conflict = new Response(null, { status: 409, statusText: 'Conflict' })
    expect(() => unwrap({ error: { detail: 'Case is closed' }, response: conflict })).toThrow(
      new ApiError(409, 'Case is closed'),
    )
  })
})
