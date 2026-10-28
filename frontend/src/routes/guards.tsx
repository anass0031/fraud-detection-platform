import { Center, Loader } from '@mantine/core'
import type { ReactNode } from 'react'
import { Navigate, Outlet, useLocation } from 'react-router'

import type { User } from '../api/client'
import { useAuth } from '../features/auth/context'

/** Logged-in users only; others go to the login page, which brings them back afterwards. */
export function RequireAuth({ children }: { children?: ReactNode }) {
  const auth = useAuth()
  const location = useLocation()
  if (auth.status === 'loading') {
    return (
      <Center h="100vh">
        <Loader aria-label="Restoring session" />
      </Center>
    )
  }
  if (auth.status === 'anonymous') {
    const next = encodeURIComponent(location.pathname + location.search)
    return <Navigate to={`/login?next=${next}`} replace />
  }
  return children ?? <Outlet />
}

/** Only the given roles; others see the 403 page. Use inside RequireAuth. */
export function RequireRole({ roles, children }: { roles: User['role'][]; children?: ReactNode }) {
  const { user } = useAuth()
  if (user === null || !roles.includes(user.role)) return <Navigate to="/403" replace />
  return children ?? <Outlet />
}
