import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getAccessToken, setAccessToken } from '../../api/client'
import { ANALYST } from '../../test/render'
import { AuthProvider } from './AuthProvider'
import { useAuth } from './context'

function Panel() {
  const auth = useAuth()
  return (
    <div>
      <p>status: {auth.status}</p>
      <button onClick={() => void auth.login('ann@example.com', 'pw-123456789').catch(() => {})}>
        login
      </button>
      <button onClick={() => void auth.logout()}>logout</button>
    </div>
  )
}

const fetchMock = vi.fn<typeof fetch>()
const calls: { path: string; body: string }[] = []
let client: QueryClient

beforeEach(() => {
  calls.length = 0
  setAccessToken(null)
  fetchMock.mockReset()
  fetchMock.mockImplementation(async (input) => {
    const req = input as Request
    const path = new URL(req.url).pathname
    calls.push({ path, body: await req.clone().text() })
    if (path.endsWith('/refresh')) return new Response('{}', { status: 401 }) // no session yet
    if (path.endsWith('/login')) {
      return new Response(
        JSON.stringify({
          access_token: 'tok',
          token_type: 'bearer',
          expires_in: 900,
          user: ANALYST,
        }),
      )
    }
    return new Response(null, { status: 204 })
  })
  vi.stubGlobal('fetch', fetchMock)
  client = new QueryClient()
  render(
    <QueryClientProvider client={client}>
      <AuthProvider>
        <Panel />
      </AuthProvider>
    </QueryClientProvider>,
  )
})
afterEach(() => vi.unstubAllGlobals())

describe('AuthProvider login / logout', () => {
  it('posts the OAuth2 form, keeps the token in memory, and logs out cleanly', async () => {
    const user = userEvent.setup()
    expect(await screen.findByText('status: anonymous')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'login' }))
    expect(await screen.findByText('status: authenticated')).toBeInTheDocument()
    const login = calls.find((c) => c.path === '/api/v1/auth/login')!
    expect(new URLSearchParams(login.body).get('username')).toBe('ann@example.com')
    expect(new URLSearchParams(login.body).get('password')).toBe('pw-123456789')
    expect(getAccessToken()).toBe('tok')

    client.setQueryData(['cases', {}], { cached: true })
    await user.click(screen.getByRole('button', { name: 'logout' }))
    await waitFor(() => expect(screen.getByText('status: anonymous')).toBeInTheDocument())
    expect(calls.some((c) => c.path === '/api/v1/auth/logout')).toBe(true)
    expect(getAccessToken()).toBeNull()
    expect(client.getQueryData(['cases', {}])).toBeUndefined() // cache cleared on logout
  })

  it('stays anonymous when the login is refused', async () => {
    fetchMock.mockImplementation(async (input) => {
      const path = new URL((input as Request).url).pathname
      return path.endsWith('/login')
        ? new Response(JSON.stringify({ detail: 'Incorrect e-mail or password' }), { status: 401 })
        : new Response('{}', { status: 401 })
    })
    await userEvent.click(screen.getByRole('button', { name: 'login' }))
    await waitFor(() => expect(screen.getByText('status: anonymous')).toBeInTheDocument())
    expect(getAccessToken()).toBeNull()
  })
})
