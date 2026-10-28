/**
 * The queue "view" (tab, filters, sort, page) <-> URL search params <-> API query.
 * The URL is the single source of truth, so reload, back button and shared links keep the view.
 */
import type { User } from '../../api/client'
import type { CaseRow, CasesQuery } from './api'

export type Tab = 'open' | 'mine' | 'all'
export type Risk = CaseRow['risk_level']
export type Reason = CaseRow['reason']
export type Status = CaseRow['status']
export type Sort = NonNullable<CasesQuery['sort']>
export type Order = 'asc' | 'desc'

export type QueueView = {
  tab: Tab
  risk: Risk[]
  reason: Reason | null
  status: Status[] // only used on the "all" tab
  amountMin: number | null
  amountMax: number | null
  dateFrom: string | null // YYYY-MM-DD, transaction date, inclusive
  dateTo: string | null
  sort: Sort
  order: Order
  page: number // 1-based
  size: number
}

export const PAGE_SIZES = [25, 50, 100] as const
export const RISKS: Risk[] = ['low', 'medium', 'high', 'critical']
export const STATUSES: Status[] = ['new', 'assigned', 'escalated', 'closed']
const TABS: Tab[] = ['open', 'mine', 'all']
const REASONS: Reason[] = ['model_score', 'anomaly']
const SORTS: Sort[] = ['priority', 'amount', 'transaction_time', 'created_at']

export const DEFAULT_VIEW: QueueView = {
  tab: 'open',
  risk: [],
  reason: null,
  status: [],
  amountMin: null,
  amountMax: null,
  dateFrom: null,
  dateTo: null,
  sort: 'priority',
  order: 'desc',
  page: 1,
  size: 50,
}

function oneOf<T extends string>(value: string | null, allowed: readonly T[]): T | null {
  return value !== null && (allowed as readonly string[]).includes(value) ? (value as T) : null
}

function manyOf<T extends string>(values: string[], allowed: readonly T[]): T[] {
  return [...new Set(values)].filter((v): v is T => (allowed as readonly string[]).includes(v))
}

function amount(value: string | null): number | null {
  if (value === null || value.trim() === '') return null
  const n = Number(value)
  return Number.isFinite(n) && n >= 0 ? n : null
}

function date(value: string | null): string | null {
  return value !== null && /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : null
}

/** Read a view from the URL. Unknown or invalid values fall back to the defaults. */
export function parseView(params: URLSearchParams): QueueView {
  const page = Number(params.get('page'))
  const size = Number(params.get('size'))
  return {
    tab: oneOf(params.get('tab'), TABS) ?? DEFAULT_VIEW.tab,
    risk: manyOf(params.getAll('risk'), RISKS),
    reason: oneOf(params.get('reason'), REASONS),
    status: manyOf(params.getAll('status'), STATUSES),
    amountMin: amount(params.get('min')),
    amountMax: amount(params.get('max')),
    dateFrom: date(params.get('from')),
    dateTo: date(params.get('to')),
    sort: oneOf(params.get('sort'), SORTS) ?? DEFAULT_VIEW.sort,
    order: oneOf(params.get('order'), ['asc', 'desc'] as const) ?? DEFAULT_VIEW.order,
    page: Number.isInteger(page) && page >= 1 ? page : 1,
    size: (PAGE_SIZES as readonly number[]).includes(size) ? size : DEFAULT_VIEW.size,
  }
}

/** Write a view to URL params; default values are left out to keep links short. */
export function viewToParams(view: QueueView): URLSearchParams {
  const p = new URLSearchParams()
  if (view.tab !== DEFAULT_VIEW.tab) p.set('tab', view.tab)
  view.risk.forEach((r) => p.append('risk', r))
  if (view.reason) p.set('reason', view.reason)
  if (view.tab === 'all') view.status.forEach((s) => p.append('status', s))
  if (view.amountMin !== null) p.set('min', String(view.amountMin))
  if (view.amountMax !== null) p.set('max', String(view.amountMax))
  if (view.dateFrom) p.set('from', view.dateFrom)
  if (view.dateTo) p.set('to', view.dateTo)
  if (view.sort !== DEFAULT_VIEW.sort) p.set('sort', view.sort)
  if (view.order !== DEFAULT_VIEW.order) p.set('order', view.order)
  if (view.page !== 1) p.set('page', String(view.page))
  if (view.size !== DEFAULT_VIEW.size) p.set('size', String(view.size))
  return p
}

/** Which statuses / assignee a tab shows. Admins also handle escalated cases. */
function tabFilter(view: QueueView, user: User): Pick<CasesQuery, 'status' | 'assignee'> {
  switch (view.tab) {
    case 'open':
      return { status: user.role === 'admin' ? ['new', 'escalated'] : ['new'] }
    case 'mine':
      return { status: ['assigned', 'escalated'], assignee: 'me' }
    case 'all':
      return view.status.length > 0 ? { status: view.status } : {}
  }
}

/** The API request for a view. Empty filters are not sent. */
export function viewToQuery(view: QueueView, user: User): CasesQuery {
  const query: CasesQuery = {
    ...tabFilter(view, user),
    sort: view.sort,
    order: view.order,
    limit: view.size,
    offset: (view.page - 1) * view.size,
  }
  if (view.risk.length > 0) query.risk_level = view.risk
  if (view.reason) query.reason = view.reason
  if (view.amountMin !== null) query.amount_min = view.amountMin
  if (view.amountMax !== null) query.amount_max = view.amountMax
  if (view.dateFrom) query.date_from = view.dateFrom
  if (view.dateTo) query.date_to = view.dateTo
  return query
}

/** A change to anything but the page sends the user back to page 1. */
export function updateView(view: QueueView, change: Partial<QueueView>): QueueView {
  const next = { ...view, ...change }
  if (!('page' in change)) next.page = 1
  return next
}

export function hasFilters(view: QueueView): boolean {
  return (
    view.risk.length > 0 ||
    view.reason !== null ||
    view.status.length > 0 ||
    view.amountMin !== null ||
    view.amountMax !== null ||
    view.dateFrom !== null ||
    view.dateTo !== null
  )
}
