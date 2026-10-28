/** Display formatters (en-US, USD: the dataset's currency). */

const usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' })

export function money(value: number): string {
  return usd.format(value)
}

/**
 * Probability as a whole percent. Calibrated probabilities are coarse at the top (few distinct
 * values), so decimals would suggest false precision. Tiny non-zero values show as "<1%".
 */
export function percent(probability: number): string {
  const pct = Math.round(probability * 100)
  if (pct === 0 && probability > 0) return '<1%'
  return `${pct}%`
}

/**
 * Transaction times are on the dataset clock (no time zone): show them exactly as stored,
 * never converted to the viewer's zone.
 */
export function datasetTime(iso: string): string {
  const [date, time = ''] = iso.split('T')
  return `${date} ${time.slice(0, 5)}`
}

/** Real timestamps (with zone), shown in the viewer's local time. */
export function localTime(iso: string): string {
  return new Date(iso).toLocaleString('en-GB', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** Short age like "5 min", "3 h", "2 d" between a timestamp and now. */
export function age(iso: string, now: Date = new Date()): string {
  const minutes = Math.max(0, Math.floor((now.getTime() - new Date(iso).getTime()) / 60_000))
  if (minutes < 60) return `${minutes} min`
  const hours = Math.floor(minutes / 60)
  if (hours < 48) return `${hours} h`
  return `${Math.floor(hours / 24)} d`
}
