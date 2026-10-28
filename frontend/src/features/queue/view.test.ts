import { describe, expect, it } from 'vitest'

import { ADMIN, ANALYST } from '../../test/render'
import { DEFAULT_VIEW, hasFilters, parseView, updateView, viewToParams, viewToQuery } from './view'

const params = (s: string) => new URLSearchParams(s)

describe('URL <-> view', () => {
  it('an empty URL is the default view, and the default view writes an empty URL', () => {
    expect(parseView(params(''))).toEqual(DEFAULT_VIEW)
    expect(viewToParams(DEFAULT_VIEW).toString()).toBe('')
  })

  it('round-trips every field', () => {
    const url =
      'tab=all&risk=high&risk=critical&reason=anomaly&status=closed&min=10&max=500.5' +
      '&from=2018-04-28&to=2018-04-29&sort=amount&order=asc&page=3&size=100'
    expect(viewToParams(parseView(params(url))).toString()).toBe(url)
  })

  it('ignores invalid or unknown values', () => {
    const view = parseView(
      params(
        'tab=x&risk=huge&risk=high&risk=high&reason=?&min=-1&max=abc&from=28/04/2018' +
          '&sort=score&order=up&page=0&size=7',
      ),
    )
    expect(view).toEqual({ ...DEFAULT_VIEW, risk: ['high'] })
  })

  it('status filter is only kept on the "all" tab', () => {
    const view = parseView(params('tab=open&status=closed'))
    expect(viewToParams(view).toString()).toBe('')
  })
})

describe('view -> API query', () => {
  it('open tab: new cases for analysts, new + escalated for admins', () => {
    expect(viewToQuery(DEFAULT_VIEW, ANALYST)).toEqual({
      status: ['new'],
      sort: 'priority',
      order: 'desc',
      limit: 50,
      offset: 0,
    })
    expect(viewToQuery(DEFAULT_VIEW, ADMIN).status).toEqual(['new', 'escalated'])
  })

  it('my cases: assigned to me and still open', () => {
    const q = viewToQuery({ ...DEFAULT_VIEW, tab: 'mine' }, ANALYST)
    expect(q.assignee).toBe('me')
    expect(q.status).toEqual(['assigned', 'escalated'])
  })

  it('all tab: any status unless filtered', () => {
    expect(viewToQuery({ ...DEFAULT_VIEW, tab: 'all' }, ANALYST).status).toBeUndefined()
    expect(
      viewToQuery({ ...DEFAULT_VIEW, tab: 'all', status: ['closed'] }, ANALYST).status,
    ).toEqual(['closed'])
  })

  it('sends filters and the page offset', () => {
    const q = viewToQuery(
      parseView(params('risk=critical&min=100&from=2018-04-28&to=2018-04-28&page=3&size=25')),
      ANALYST,
    )
    expect(q).toMatchObject({
      risk_level: ['critical'],
      amount_min: 100,
      date_from: '2018-04-28',
      date_to: '2018-04-28',
      limit: 25,
      offset: 50,
    })
    expect(q).not.toHaveProperty('amount_max')
    expect(q).not.toHaveProperty('reason')
  })
})

describe('updateView', () => {
  it('any change but the page goes back to page 1', () => {
    const onPage3 = { ...DEFAULT_VIEW, page: 3 }
    expect(updateView(onPage3, { risk: ['high'] }).page).toBe(1)
    expect(updateView(onPage3, { page: 4 }).page).toBe(4)
  })

  it('knows when filters are set', () => {
    expect(hasFilters(DEFAULT_VIEW)).toBe(false)
    expect(hasFilters({ ...DEFAULT_VIEW, amountMax: 10 })).toBe(true)
  })
})
