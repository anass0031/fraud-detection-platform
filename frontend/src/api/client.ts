/**
 * Typed API client (openapi-fetch + types generated from the backend's OpenAPI spec).
 *
 * Auth model: the access token lives only in memory (this module). The refresh token is an
 * httpOnly cookie the page cannot read. A request that gets 401 triggers ONE shared refresh,
 * then is retried once. Sharing matters: the backend treats a refresh token used twice as
 * stolen and revokes every session of the user.
 */
import createClient from 'openapi-fetch'

import type { components, paths } from './schema'

export type Schemas = components['schemas']
export type User = Schemas['UserOut']
export type TokenResponse = Schemas['TokenResponse']

const AUTH_PREFIX = '/api/v1/auth/'

// ---------------------------------------------------------------- token store

let accessToken: string | null = null
const expiredListeners = new Set<() => void>()

export function getAccessToken(): string | null {
  return accessToken
}

export function setAccessToken(token: string | null): void {
  accessToken = token
}

/** Called when the session cannot be renewed (refresh failed): the UI shows the login page. */
export function onSessionExpired(listener: () => void): () => void {
  expiredListeners.add(listener)
  return () => expiredListeners.delete(listener)
}

// ---------------------------------------------------------------- refresh (single flight)

let refreshing: Promise<TokenResponse | null> | null = null

/**
 * Swap the refresh cookie for a new access token. Concurrent callers share the same request.
 * Resolves to null when there is no valid session (no cookie, expired, revoked).
 */
export function refreshSession(): Promise<TokenResponse | null> {
  if (refreshing === null) {
    refreshing = fetch(`${AUTH_PREFIX}refresh`, { method: 'POST', credentials: 'same-origin' })
      .then(async (resp) => (resp.ok ? ((await resp.json()) as TokenResponse) : null))
      .catch(() => null)
      .then((session) => {
        setAccessToken(session?.access_token ?? null)
        return session
      })
      .finally(() => {
        refreshing = null
      })
  }
  return refreshing
}

// ---------------------------------------------------------------- fetch with auth

function withToken(request: Request, token: string | null): Request {
  const headers = new Headers(request.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return new Request(request, { headers })
}

/** fetch for the API: adds the Bearer token; on 401 refreshes once and retries once. */
export async function authFetch(request: Request): Promise<Response> {
  const retry = request.clone() // a request body can only be read once
  const resp = await fetch(withToken(request, accessToken))
  const isAuthCall = new URL(request.url, window.location.origin).pathname.startsWith(AUTH_PREFIX)
  if (resp.status !== 401 || isAuthCall) return resp

  const session = await refreshSession()
  if (session === null) {
    expiredListeners.forEach((listener) => listener())
    return resp
  }
  return fetch(withToken(retry, session.access_token))
}

export const api = createClient<paths>({
  baseUrl: window.location.origin,
  fetch: authFetch,
  credentials: 'same-origin',
})

// ---------------------------------------------------------------- errors

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** The `detail` message of a FastAPI error body (string, or the first validation error). */
export function errorMessage(error: unknown, fallback = 'Something went wrong'): string {
  if (error instanceof ApiError) return error.message
  const detail = (error as { detail?: unknown } | undefined)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: string }
    if (first.msg) return first.msg
  }
  return error instanceof Error ? error.message : fallback
}

/** Unwrap an openapi-fetch result: the data, or throw ApiError with the server's message. */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.error !== undefined || result.data === undefined) {
    throw new ApiError(
      result.response.status,
      errorMessage(result.error, result.response.statusText),
    )
  }
  return result.data
}
