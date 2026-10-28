import { useQueryClient } from '@tanstack/react-query'
import { type ReactNode, useCallback, useEffect, useMemo, useState } from 'react'

import {
  type User,
  api,
  onSessionExpired,
  refreshSession,
  setAccessToken,
  unwrap,
} from '../../api/client'
import { AuthContext, type AuthState } from './context'

/**
 * Holds the logged-in user. On start-up the session is restored from the refresh cookie
 * (one shared refresh, so React StrictMode's double effects are harmless).
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  const [state, setState] = useState<AuthState>({ status: 'loading', user: null })

  const signedOut = useCallback(() => {
    setAccessToken(null)
    queryClient.clear()
    setState({ status: 'anonymous', user: null })
  }, [queryClient])

  useEffect(() => {
    let active = true
    void refreshSession().then((session) => {
      if (!active) return
      setState(
        session
          ? { status: 'authenticated', user: session.user }
          : { status: 'anonymous', user: null },
      )
    })
    const unsubscribe = onSessionExpired(signedOut)
    return () => {
      active = false
      unsubscribe()
    }
  }, [signedOut])

  const login = useCallback(async (email: string, password: string): Promise<User> => {
    const session = unwrap(
      await api.POST('/api/v1/auth/login', {
        body: { username: email, password, scope: '' },
        bodySerializer: (body) => new URLSearchParams(body as Record<string, string>),
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      }),
    )
    setAccessToken(session.access_token)
    setState({ status: 'authenticated', user: session.user })
    return session.user
  }, [])

  const logout = useCallback(async () => {
    try {
      await api.POST('/api/v1/auth/logout')
    } finally {
      signedOut()
    }
  }, [signedOut])

  const value = useMemo(() => ({ ...state, login, logout }), [state, login, logout])
  return <AuthContext value={value}>{children}</AuthContext>
}
