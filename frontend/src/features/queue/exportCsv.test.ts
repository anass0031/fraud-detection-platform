import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { downloadCasesCsv, exportQuery, fileName } from './exportCsv'

describe('exportQuery', () => {
  it('keeps the filters and drops sorting and paging', () => {
    expect(
      exportQuery({
        status: ['new'],
        risk_level: ['critical'],
        amount_min: 100,
        sort: 'amount',
        order: 'asc',
        limit: 50,
        offset: 100,
      }),
    ).toEqual({ status: ['new'], risk_level: ['critical'], amount_min: 100 })
  })
})

describe('fileName', () => {
  it.each([
    ['attachment; filename="cases-20261028.csv"', 'cases-20261028.csv'],
    ['attachment; filename=cases.csv', 'cases.csv'],
    [null, 'cases.csv'],
  ])('%s -> %s', (header, name) => {
    expect(fileName(header)).toBe(name)
  })
})

describe('downloadCasesCsv', () => {
  const fetchMock = vi.fn<typeof fetch>()
  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock)
    URL.createObjectURL = vi.fn(() => 'blob:x')
    URL.revokeObjectURL = vi.fn()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('requests the filtered CSV and saves it under the server name', async () => {
    fetchMock.mockResolvedValue(
      new Response('case_id\n1\n', {
        headers: {
          'Content-Type': 'text/csv',
          'Content-Disposition': 'attachment; filename="cases-20261028.csv"',
        },
      }),
    )
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    const name = await downloadCasesCsv({ status: ['closed'], limit: 50, offset: 0 })
    expect(name).toBe('cases-20261028.csv')
    const url = new URL((fetchMock.mock.calls[0][0] as Request).url)
    expect(url.pathname).toBe('/api/v1/reports/cases.csv')
    expect(url.searchParams.getAll('status')).toEqual(['closed'])
    expect(url.searchParams.has('limit')).toBe(false)
    expect(click).toHaveBeenCalledOnce()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:x')
  })

  it('reports a server error', async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Not authenticated' }), { status: 401 }),
    )
    await expect(downloadCasesCsv({})).rejects.toThrow('Not authenticated')
  })
})
