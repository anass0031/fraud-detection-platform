import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { ADMIN, ANALYST, fakeAuth, renderRoutes } from '../test/render'
import { AppLayout } from './AppLayout'

const routes = [
  {
    path: '/',
    element: <AppLayout />,
    children: [
      { path: 'queue', element: <p>queue page</p> },
      { path: 'monitoring', element: <p>monitoring page</p> },
    ],
  },
]

function navLabels() {
  return within(screen.getByRole('navigation'))
    .getAllByRole('link')
    .map((a) => a.textContent)
}

describe('AppLayout', () => {
  it('analysts see the queue, their cases and monitoring only', () => {
    renderRoutes(routes, '/queue', fakeAuth(ANALYST))
    expect(navLabels()).toEqual(['Review queue', 'My cases', 'Monitoring'])
    expect(screen.getByText('Ann Analyst')).toBeInTheDocument()
    expect(screen.getByText('analyst')).toBeInTheDocument()
  })

  it('admins also see the models and the threshold page', () => {
    renderRoutes(routes, '/queue', fakeAuth(ADMIN))
    expect(navLabels()).toEqual([
      'Review queue',
      'My cases',
      'Monitoring',
      'Models',
      'Threshold and costs',
    ])
  })

  it('marks the current page and logs out', async () => {
    const logout = vi.fn(async () => {})
    renderRoutes(routes, '/queue?tab=mine', fakeAuth(ANALYST, { logout }))
    const nav = within(screen.getByRole('navigation'))
    expect(nav.getByRole('link', { name: 'My cases' })).toHaveAttribute('data-active', 'true')
    expect(nav.getByRole('link', { name: 'Review queue' })).not.toHaveAttribute('data-active')
    await userEvent.click(screen.getByRole('button', { name: 'Log out' }))
    expect(logout).toHaveBeenCalledOnce()
  })
})
