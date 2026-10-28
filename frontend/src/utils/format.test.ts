import { describe, expect, it } from 'vitest'

import { age, datasetTime, money, percent } from './format'

describe('formatters', () => {
  it('money in USD', () => {
    expect(money(994)).toBe('$994.00')
    expect(money(480.8878)).toBe('$480.89')
    expect(money(1234567.5)).toBe('$1,234,567.50')
  })

  it('percent as a whole number, tiny values as <1%', () => {
    expect(percent(0.4837)).toBe('48%')
    expect(percent(0.995)).toBe('100%')
    expect(percent(0.004)).toBe('<1%')
    expect(percent(0)).toBe('0%')
  })

  it('dataset time is shown as stored (no time-zone conversion)', () => {
    expect(datasetTime('2018-04-28T19:52:25')).toBe('2018-04-28 19:52')
  })

  it('age in minutes, hours, then days', () => {
    const now = new Date('2026-10-28T12:00:00Z')
    expect(age('2026-10-28T11:55:00Z', now)).toBe('5 min')
    expect(age('2026-10-28T09:00:00Z', now)).toBe('3 h')
    expect(age('2026-10-25T12:00:00Z', now)).toBe('3 d')
    expect(age('2026-10-28T12:01:00Z', now)).toBe('0 min') // clock skew: never negative
  })
})
