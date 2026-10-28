import { screen } from '@testing-library/react'
import { useLocation } from 'react-router'
import { describe, expect, it } from 'vitest'

import { ADMIN, ANALYST, fakeAuth, renderRoutes } from '../test/render'
import { RequireAuth, RequireRole } from './guards'

function Where() {
  const loc = useLocation()
  return <p>at {loc.pathname + loc.search}</p>
}

const routes = [
  { path: '/login', element: <Where /> },
  { path: '/403', element: <p>forbidden</p> },
  {
    path: '/',
    element: <RequireAuth />,
    children: [
      { path: 'queue', element: <p>queue page</p> },
      {
        path: 'admin',
        element: <RequireRole roles={['admin']} />,
        children: [{ path: 'settings', element: <p>admin page</p> }],
      },
    ],
  },
]

describe('RequireAuth', () => {
  it('sends anonymous users to login and remembers where they wanted to go', async () => {
    const { router } = renderRoutes(routes, '/queue?tab=mine', fakeAuth(null))
    expect(await screen.findByText(/^at \/login/)).toBeInTheDocument()
    expect(router.state.location.search).toBe('?next=%2Fqueue%3Ftab%3Dmine')
  })

  it('waits while the session is being restored', () => {
    renderRoutes(routes, '/queue', fakeAuth('loading'))
    expect(screen.getByLabelText('Restoring session')).toBeInTheDocument()
    expect(screen.queryByText('queue page')).not.toBeInTheDocument()
  })

  it('shows the page to logged-in users', () => {
    renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    expect(screen.getByText('queue page')).toBeInTheDocument()
  })
})

describe('RequireRole', () => {
  it('shows 403 to an analyst on an admin page', async () => {
    renderRoutes(routes, '/admin/settings', fakeAuth(ANALYST))
    expect(await screen.findByText('forbidden')).toBeInTheDocument()
  })

  it('lets an admin in', () => {
    renderRoutes(routes, '/admin/settings', fakeAuth(ADMIN))
    expect(screen.getByText('admin page')).toBeInTheDocument()
  })
})
