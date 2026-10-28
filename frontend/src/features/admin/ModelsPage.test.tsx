import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ADMIN, fakeAuth, renderWith } from '../../test/render'
import { CANDIDATE, CHAMPION } from './models.test-data'
import { ModelsPage } from './ModelsPage'

const fetchMock = vi.fn<typeof fetch>()
const posts: string[] = []

beforeEach(() => {
  posts.length = 0
  fetchMock.mockReset()
  fetchMock.mockImplementation(async (input) => {
    const req = input as Request
    const path = new URL(req.url).pathname
    if (req.method === 'POST') {
      posts.push(path)
      if (path.endsWith('/activate')) {
        return new Response(
          JSON.stringify({
            model: { ...CANDIDATE, status: 'active', serving: true },
            threshold_override_reset: true,
          }),
        )
      }
      return new Response(JSON.stringify({ ...CANDIDATE, status: 'archived' }))
    }
    return new Response(JSON.stringify([CHAMPION, CANDIDATE]))
  })
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

describe('ModelsPage', () => {
  it('lists versions and compares the candidate with the current model', async () => {
    renderWith(<ModelsPage />, fakeAuth(ADMIN))
    const champion = await screen.findByRole('row', { name: 'lgbm-2026.10.27-1' })
    expect(within(champion).getByText('active')).toBeInTheDocument()
    expect(within(champion).getByText('serving')).toBeInTheDocument()
    expect(within(champion).getByText('2017-12-02 → 2018-03-12')).toBeInTheDocument()
    expect(within(champion).queryByRole('button', { name: 'Activate' })).not.toBeInTheDocument()

    const candidate = screen.getByRole('row', { name: 'lgbm-2026.10.28-1' })
    expect(within(candidate).getByText('$17.92')).toBeInTheDocument()
    expect(within(candidate).getByRole('button', { name: 'Archive' })).toBeInTheDocument()

    const card = screen.getByLabelText('Comparison for lgbm-2026.10.28-1')
    const table = within(card).getByRole('table', { name: 'Champion vs challenger' })
    const prAuc = within(table).getByText('PR-AUC').closest('tr')!
    expect(within(prAuc).getByText('0.432')).toBeInTheDocument()
    expect(within(prAuc).getByText('0.532')).toHaveAttribute('data-better', 'yes')
    expect(within(card).getByText(/the challenger is better beyond chance/)).toBeInTheDocument()

    await userEvent.click(within(card).getByText('Excluding past-fraud customers'))
    expect(within(table).getByText('0.519')).toBeInTheDocument()
    expect(screen.getAllByText(/register_model/).length).toBeGreaterThan(0)
  })

  it('activates only after confirming, and explains the consequences', async () => {
    const user = userEvent.setup()
    renderWith(<ModelsPage />, fakeAuth(ADMIN))
    const candidate = await screen.findByRole('row', { name: 'lgbm-2026.10.28-1' })
    await user.click(within(candidate).getByRole('button', { name: 'Activate' }))

    expect(await screen.findByText('Activate lgbm-2026.10.28-1?')).toBeInTheDocument()
    expect(screen.getByText(/lgbm-2026\.10\.27-1 is archived/)).toBeInTheDocument()
    expect(screen.getByText(/reset to this model’s own threshold \(\$17\.92\)/)).toBeInTheDocument()
    expect(posts).toEqual([])

    const dialog = screen.getByRole('dialog')
    await user.click(within(dialog).getByRole('button', { name: 'Activate' }))
    await waitFor(() => expect(posts).toEqual(['/api/v1/models/2/activate']))
    expect(await screen.findByText('lgbm-2026.10.28-1 is now active')).toBeInTheDocument()
    expect(screen.getByText(/custom alert threshold was reset/)).toBeInTheDocument()
  })

  it('archives a candidate', async () => {
    renderWith(<ModelsPage />, fakeAuth(ADMIN))
    const candidate = await screen.findByRole('row', { name: 'lgbm-2026.10.28-1' })
    await userEvent.click(within(candidate).getByRole('button', { name: 'Archive' }))
    await waitFor(() => expect(posts).toEqual(['/api/v1/models/2/archive']))
    expect(await screen.findByText('lgbm-2026.10.28-1 archived')).toBeInTheDocument()
  })

  it('shows an error with retry', async () => {
    fetchMock.mockImplementation(
      async () =>
        new Response(JSON.stringify({ detail: 'Not allowed for your role' }), { status: 403 }),
    )
    renderWith(<ModelsPage />, fakeAuth(ADMIN))
    expect(await screen.findByText('Not allowed for your role')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })
})
