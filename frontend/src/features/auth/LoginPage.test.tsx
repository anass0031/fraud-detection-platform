import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../api/client'
import { ANALYST, fakeAuth, renderRoutes } from '../../test/render'
import { safeNext } from '../../utils/redirect'
import { LoginPage } from './LoginPage'

const routes = [
  { path: '/login', element: <LoginPage /> },
  { path: '/queue', element: <p>queue page</p> },
  { path: '/cases/:id', element: <p>case page</p> },
]

async function fillAndSubmit(email: string, password: string) {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText(/e-mail/i), email)
  await user.type(screen.getByLabelText(/password/i, { selector: 'input' }), password)
  await user.click(screen.getByRole('button', { name: /sign in/i }))
}

describe('LoginPage', () => {
  it('logs in and goes back to the page that was asked for', async () => {
    const login = vi.fn().mockResolvedValue(ANALYST)
    const { router } = renderRoutes(routes, '/login?next=%2Fcases%2F7', fakeAuth(null, { login }))
    await fillAndSubmit('  ann@example.com ', 'secret-password')
    expect(login).toHaveBeenCalledWith('ann@example.com', 'secret-password')
    await waitFor(() => expect(router.state.location.pathname).toBe('/cases/7'))
  })

  it('shows the server message on failure and stays on the page', async () => {
    const login = vi.fn().mockRejectedValue(new ApiError(401, 'Incorrect e-mail or password'))
    renderRoutes(routes, '/login', fakeAuth(null, { login }))
    await fillAndSubmit('ann@example.com', 'wrong-password')
    expect(await screen.findByRole('alert')).toHaveTextContent('Incorrect e-mail or password')
    expect(screen.getByRole('button', { name: /sign in/i })).toBeEnabled()
  })

  it('disables the button until both fields are filled', () => {
    renderRoutes(routes, '/login', fakeAuth(null))
    expect(screen.getByRole('button', { name: /sign in/i })).toBeDisabled()
  })

  it('redirects a logged-in user straight to the queue', async () => {
    const { router } = renderRoutes(routes, '/login', fakeAuth(ANALYST))
    await waitFor(() => expect(router.state.location.pathname).toBe('/queue'))
  })
})

describe('safeNext', () => {
  it.each([
    ['/cases/3', '/cases/3'],
    [null, '/queue'],
    ['https://evil.example', '/queue'],
    ['//evil.example', '/queue'],
    ['/\\evil.example', '/queue'],
  ])('%s gives %s', (next, expected) => {
    expect(safeNext(next)).toBe(expected)
  })
})
