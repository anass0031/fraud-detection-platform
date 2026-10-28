import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getAccessToken, setAccessToken } from '../../api/client'
import { ANALYST } from '../../test/render'
import { AuthProvider } from './AuthProvider'
import { useAuth } from './context'

function Status() {
  const auth = useAuth()
  return <p>{auth.status === 'authenticated' ? `hello ${auth.user.full_name}` : auth.status}</p>
}

function renderProvider() {
  return render(
    <StrictMode>
      <QueryClientProvider client={new QueryClient()}>
        <AuthProvider>
          <Status />
        </AuthProvider>
      </QueryClientProvider>
    </StrictMode>,
  )
}

const fetchMock = vi.fn<typeof fetch>()

beforeEach(() => {
  setAccessToken(null)
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

describe('AuthProvider', () => {
  it('restores the session with ONE refresh call, even with StrictMode double effects', async () => {
    const body = { access_token: 'tok', token_type: 'bearer', expires_in: 900, user: ANALYST }
    fetchMock.mockResolvedValue(new Response(JSON.stringify(body), { status: 200 }))
    renderProvider()
    expect(await screen.findByText('hello Ann Analyst')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(getAccessToken()).toBe('tok')
  })

  it('is anonymous when there is no session', async () => {
    fetchMock.mockResolvedValue(new Response('{}', { status: 401 }))
    renderProvider()
    expect(await screen.findByText('anonymous')).toBeInTheDocument()
  })
})
