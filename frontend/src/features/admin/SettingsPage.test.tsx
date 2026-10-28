import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ADMIN, fakeAuth, renderWith } from '../../test/render'
import { SettingsPage } from './SettingsPage'
import { bandsError, costsError, thresholdError } from './settingsApi'

const THRESHOLDS = {
  alert_threshold: null,
  risk_bands: [0.05, 0.2, 0.5],
  effective_threshold: 21.82,
  model_version: 'lgbm-2026.10.27-1',
  model_threshold: 21.82,
  alert_policy: 'expected_loss',
  updated_by: null,
  updated_at: null,
}
const COSTS = {
  review_cost: 5,
  friction_cost: 2,
  capacity_per_day: 150,
  updated_by: { id: 2, full_name: 'Ada Admin', email: 'a@x.io' },
  updated_at: '2026-10-28T10:00:00Z',
}

const fetchMock = vi.fn<typeof fetch>()
const puts: { path: string; body: unknown }[] = []
beforeEach(() => {
  puts.length = 0
  fetchMock.mockReset()
  fetchMock.mockImplementation(async (input) => {
    const req = input as Request
    const path = new URL(req.url).pathname
    if (req.method === 'PUT') {
      const body = await req.json()
      puts.push({ path, body })
      return new Response(
        JSON.stringify({ ...(path.endsWith('costs') ? COSTS : THRESHOLDS), ...body }),
      )
    }
    return new Response(JSON.stringify(path.endsWith('costs') ? COSTS : THRESHOLDS))
  })
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => vi.unstubAllGlobals())

describe('validation (same rules as the API)', () => {
  it.each([
    [[5, 20, 50], null],
    [[0, 20, 50], 'Each bound must be between 0% and 100%'],
    [[5, 20, 100], 'Each bound must be between 0% and 100%'],
    [[20, 5, 50], 'Bounds must increase: low < medium < high'],
    [[5, 5, 50], 'Bounds must increase: low < medium < high'],
    [['', 20, 50], 'Each bound must be between 0% and 100%'],
  ])('bands %j -> %s', (bands, expected) => {
    expect(bandsError(bands)).toBe(expected)
  })

  it('threshold only checked when custom', () => {
    expect(thresholdError(false, '')).toBeNull()
    expect(thresholdError(true, '')).toBe('Enter an amount above $0')
    expect(thresholdError(true, 0)).toBe('Enter an amount above $0')
    expect(thresholdError(true, 250_000)).toBe('At most $100,000')
    expect(thresholdError(true, 25)).toBeNull()
  })

  it('costs', () => {
    expect(costsError({ review: 5, friction: 2, capacity: 150 })).toBeNull()
    expect(costsError({ review: -1, friction: 2, capacity: 150 })).toMatch(/Review cost/)
    expect(costsError({ review: 5, friction: '', capacity: 150 })).toMatch(/Friction cost/)
    expect(costsError({ review: 5, friction: 2, capacity: 0 })).toMatch(/Capacity/)
    expect(costsError({ review: 5, friction: 2, capacity: 1.5 })).toMatch(/Capacity/)
  })
})

describe('SettingsPage', () => {
  it('shows the threshold in force, the bands, the costs and the cost curve', async () => {
    renderWith(<SettingsPage />, fakeAuth(ADMIN))
    expect(await screen.findByText(/In force now/)).toHaveTextContent(
      'In force now: $21.82 (model lgbm-2026.10.27-1, own threshold $21.82)',
    )
    expect(screen.getByRole('radio', { name: /Use the model’s own threshold/ })).toBeChecked()
    expect(screen.getByLabelText('Low below')).toHaveValue('5%')
    expect(screen.getByLabelText('Review capacity (cases per day)')).toHaveValue('150')
    expect(screen.getByText(/Last changed .* by Ada Admin/)).toBeInTheDocument()
    expect(
      screen.getByRole('img', { name: /Total cost versus mean alerts per day/ }),
    ).toHaveAttribute('src', '/figures/model_cost_policies.png')
  })

  it('saves a custom threshold with the bands as fractions', async () => {
    const user = userEvent.setup()
    renderWith(<SettingsPage />, fakeAuth(ADMIN))
    await user.click(await screen.findByRole('radio', { name: 'Custom threshold' }))
    const input = screen.getByLabelText(/Custom threshold \(expected loss/)
    await user.clear(input)
    await user.type(input, '30')
    await user.click(screen.getByRole('button', { name: 'Save thresholds' }))
    await waitFor(() =>
      expect(puts).toEqual([
        {
          path: '/api/v1/settings/thresholds',
          body: { alert_threshold: 30, risk_bands: [0.05, 0.2, 0.5] },
        },
      ]),
    )
    expect(await screen.findByText(/Threshold settings saved/)).toBeInTheDocument()
  })

  it('blocks saving invalid bands', async () => {
    const user = userEvent.setup()
    renderWith(<SettingsPage />, fakeAuth(ADMIN))
    const medium = await screen.findByLabelText('Medium below')
    await user.clear(medium)
    await user.type(medium, '3')
    expect(screen.getByText('Bounds must increase: low < medium < high')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Save thresholds' })).toBeDisabled()
  })

  it('saves the costs', async () => {
    const user = userEvent.setup()
    renderWith(<SettingsPage />, fakeAuth(ADMIN))
    const capacity = await screen.findByLabelText('Review capacity (cases per day)')
    await user.clear(capacity)
    await user.type(capacity, '200')
    await user.click(screen.getByRole('button', { name: 'Save costs' }))
    await waitFor(() =>
      expect(puts).toEqual([
        {
          path: '/api/v1/settings/costs',
          body: { review_cost: 5, friction_cost: 2, capacity_per_day: 200 },
        },
      ]),
    )
  })
})
