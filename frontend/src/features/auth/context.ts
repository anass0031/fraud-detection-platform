import { createContext, use } from 'react'

import type { User } from '../../api/client'

export type AuthState =
  | { status: 'loading'; user: null }
  | { status: 'anonymous'; user: null }
  | { status: 'authenticated'; user: User }

export type Auth = AuthState & {
  login: (email: string, password: string) => Promise<User>
  logout: () => Promise<void>
}

export const AuthContext = createContext<Auth | null>(null)

export function useAuth(): Auth {
  const auth = use(AuthContext)
  if (auth === null) throw new Error('useAuth must be used inside <AuthProvider>')
  return auth
}
