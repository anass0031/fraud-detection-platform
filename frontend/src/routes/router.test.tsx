import { screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ANALYST, fakeAuth, renderRoutes } from '../test/render'
import { RouteErrorPage } from './ErrorPages'
import { routes } from './router'

function findRoute(path: string) {
  const app = routes.find((r) => r.path === '/')!
  const direct = app.children!.find((r) => r.path === path)
  if (direct) return direct
  const [parent, child] = path.split('/') // nested, e.g. admin/models
  return app.children!.find((r) => r.path === parent)!.children!.find((r) => r.path === child)!
}

describe('routes', () => {
  it.each([
    ['queue', 'QueuePage'],
    ['cases/:caseId', 'CasePage'],
    ['monitoring', 'MonitoringPage'],
    ['admin/settings', 'SettingsPage'],
    ['admin/models', 'ModelsPage'],
  ])('%s loads its page on demand', async (path, name) => {
    const route = findRoute(path)
    expect(route.element).toBeUndefined()
    const loaded = await (route.lazy as () => Promise<{ Component: { name: string } }>)()
    expect(loaded.Component.name).toBe(name)
  })
})

describe('RouteErrorPage', () => {
  it('replaces a crashed page with a message and a reload button', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {}) // React logs the crash
    function Boom(): never {
      throw new Error('Failed to fetch dynamically imported module')
    }
    renderRoutes(
      [{ path: '/x', element: <Boom />, errorElement: <RouteErrorPage /> }],
      '/x',
      fakeAuth(ANALYST),
    )
    expect(await screen.findByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText('Failed to fetch dynamically imported module')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reload the page' })).toBeInTheDocument()
  })
})
