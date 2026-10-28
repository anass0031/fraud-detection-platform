import { MantineProvider } from '@mantine/core'
import { Notifications } from '@mantine/notifications'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { RouterProvider, createMemoryRouter, type RouteObject } from 'react-router'

import type { User } from '../api/client'
import { type Auth, AuthContext } from '../features/auth/context'

export const ANALYST: User = {
  id: 1,
  email: 'analyst@example.com',
  full_name: 'Ann Analyst',
  role: 'analyst',
  is_active: true,
  created_at: '2026-10-28T10:00:00Z',
  last_login_at: null,
}

export const ADMIN: User = {
  ...ANALYST,
  id: 2,
  email: 'admin@example.com',
  full_name: 'Ada Admin',
  role: 'admin',
}

/** A fixed auth value (no network): pass a user, null (anonymous) or 'loading'. */
export function fakeAuth(user: User | null | 'loading', overrides: Partial<Auth> = {}): Auth {
  const base =
    user === 'loading'
      ? { status: 'loading' as const, user: null }
      : user === null
        ? { status: 'anonymous' as const, user: null }
        : { status: 'authenticated' as const, user }
  return { ...base, login: async () => ANALYST, logout: async () => {}, ...overrides } as Auth
}

export function renderWith(ui: ReactElement, auth: Auth) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    // env="test": no transitions and no portals, so menus/modals are visible at once.
    <MantineProvider env="test">
      <Notifications />
      <QueryClientProvider client={client}>
        <AuthContext value={auth}>{ui}</AuthContext>
      </QueryClientProvider>
    </MantineProvider>,
  )
}

/** Render routes at a start URL; returns the router to inspect the final location. */
export function renderRoutes(routes: RouteObject[], url: string, auth: Auth) {
  const router = createMemoryRouter(routes, { initialEntries: [url] })
  const result = renderWith(<RouterProvider router={router} />, auth)
  return { ...result, router }
}
