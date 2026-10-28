import { ApiError, api, errorMessage } from '../../api/client'
import type { CasesQuery } from './api'

/** The CSV uses the queue's filters; it is always in queue order and not paged. */
export function exportQuery(q: CasesQuery) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { sort, order, limit, offset, ...filters } = q
  return filters
}

/** File name from the server's Content-Disposition, or a sensible default. */
export function fileName(disposition: string | null): string {
  const match = disposition?.match(/filename="?([^";]+)"?/)
  return match?.[1] ?? 'cases.csv'
}

/** Download the cases CSV with the logged-in user's token (a plain link cannot send it). */
export async function downloadCasesCsv(q: CasesQuery): Promise<string> {
  const { data, error, response } = await api.GET('/api/v1/reports/cases.csv', {
    params: { query: exportQuery(q) },
    parseAs: 'blob',
  })
  if (error !== undefined || data == null) {
    throw new ApiError(response.status, errorMessage(error, 'Export failed'))
  }
  const name = fileName(response.headers.get('Content-Disposition'))
  const url = URL.createObjectURL(data)
  const link = document.createElement('a')
  link.href = url
  link.download = name
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
  return name
}
