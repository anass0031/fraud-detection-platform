import { Button, Group, MultiSelect, NumberInput, Select } from '@mantine/core'
import { DatePickerInput } from '@mantine/dates'
import { useDebouncedCallback } from '@mantine/hooks'
import { useState } from 'react'

import {
  type QueueView,
  RISKS,
  STATUSES,
  hasFilters,
  type Reason,
  type Risk,
  type Status,
} from './view'

type Props = {
  view: QueueView
  onChange: (change: Partial<QueueView>) => void
  onClear: () => void
}

const REASON_OPTIONS = [
  { value: 'model_score', label: 'Model score' },
  { value: 'anomaly', label: 'Unusual pattern' },
]

function toNumber(value: string | number): number | null {
  return value === '' || value === null ? null : Number(value)
}

/** An amount box that only updates the view (and the request) when typing pauses. */
function AmountInput({
  label,
  value,
  onCommit,
}: {
  label: string
  value: number | null
  onCommit: (value: number | null) => void
}) {
  const [text, setText] = useState<string | number>(value ?? '')
  // Follow outside changes (Clear filters, back button) without an effect.
  const [shown, setShown] = useState(value)
  if (value !== shown) {
    setShown(value)
    setText(value ?? '')
  }
  const commit = useDebouncedCallback(onCommit, 500)
  return (
    <NumberInput
      label={label}
      placeholder="Any"
      prefix="$"
      min={0}
      thousandSeparator=","
      w={120}
      value={text}
      onChange={(v) => {
        setText(v)
        commit(toNumber(v))
      }}
    />
  )
}

export function FilterBar({ view, onChange, onClear }: Props) {
  return (
    <Group align="flex-end" gap="sm" wrap="wrap">
      {view.tab === 'all' && (
        <MultiSelect
          label="Status"
          placeholder={view.status.length ? undefined : 'Any'}
          data={STATUSES}
          value={view.status}
          onChange={(v) => onChange({ status: v as Status[] })}
          w={220}
          clearable
        />
      )}
      <MultiSelect
        label="Risk level"
        placeholder={view.risk.length ? undefined : 'Any'}
        data={RISKS}
        value={view.risk}
        onChange={(v) => onChange({ risk: v as Risk[] })}
        w={220}
        clearable
      />
      <Select
        label="Reason"
        placeholder="Any"
        data={REASON_OPTIONS}
        value={view.reason}
        onChange={(v) => onChange({ reason: (v as Reason | null) ?? null })}
        w={160}
        clearable
      />
      <AmountInput
        label="Min amount"
        value={view.amountMin}
        onCommit={(v) => onChange({ amountMin: v })}
      />
      <AmountInput
        label="Max amount"
        value={view.amountMax}
        onCommit={(v) => onChange({ amountMax: v })}
      />
      <DatePickerInput
        type="range"
        label="Transaction date"
        placeholder="Any"
        value={[view.dateFrom, view.dateTo]}
        onChange={([from, to]) => onChange({ dateFrom: from, dateTo: to })}
        valueFormat="YYYY-MM-DD"
        allowSingleDateInRange
        clearable
        w={230}
      />
      {hasFilters(view) && (
        <Button variant="subtle" onClick={onClear}>
          Clear filters
        </Button>
      )}
    </Group>
  )
}
