import { Group, Paper, Stack, Text } from '@mantine/core'
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  type TooltipContentProps,
  XAxis,
  YAxis,
} from 'recharts'

import type { Waterfall, WaterfallStep } from './waterfall'

const COLOR: Record<WaterfallStep['kind'], string> = {
  raises: 'var(--mantine-color-red-6)',
  lowers: 'var(--mantine-color-teal-6)',
  other: 'var(--mantine-color-gray-5)',
}

const ROW_HEIGHT = 30

function StepTooltip({ active, payload }: TooltipContentProps) {
  const step = active ? (payload?.[0]?.payload as WaterfallStep | undefined) : undefined
  if (!step) return null
  const sign = step.contribution >= 0 ? '+' : '−'
  return (
    <Paper withBorder shadow="sm" p="xs" maw={280}>
      <Text size="sm">{step.text}</Text>
      <Text size="xs" c="dimmed" mt={4}>
        Contribution {sign}
        {Math.abs(step.contribution).toFixed(2)} (log-odds)
      </Text>
    </Paper>
  )
}

/**
 * Horizontal SHAP waterfall: each bar starts where the previous one ended. The x-axis has no
 * numbers on purpose (log-odds mean nothing to analysts); the tooltip has them.
 */
export function WaterfallChart({
  waterfall,
  fixedWidth,
}: {
  waterfall: Waterfall
  /** Fixed pixel width instead of filling the container (tests: jsdom measures 0 px). */
  fixedWidth?: number
}) {
  const { steps, base, final, domain } = waterfall
  const height = steps.length * ROW_HEIGHT + 60

  const chart = (
    <BarChart
      width={fixedWidth}
      height={fixedWidth ? height : undefined}
      data={steps}
      layout="vertical"
      margin={{ top: 24, right: 16, bottom: 4, left: 8 }}
      barCategoryGap={4}
    >
      <XAxis type="number" domain={domain} hide />
      <YAxis
        type="category"
        dataKey="label"
        width={170}
        tickLine={false}
        axisLine={false}
        tick={{ fontSize: 12, fill: 'var(--mantine-color-text)' }}
      />
      <Tooltip content={StepTooltip} cursor={{ fill: 'var(--mantine-color-default-hover)' }} />
      <ReferenceLine
        x={base}
        stroke="var(--mantine-color-gray-6)"
        strokeDasharray="4 3"
        label={{
          value: 'Typical',
          position: 'top',
          fontSize: 11,
          fill: 'var(--mantine-color-dimmed)',
        }}
      />
      <ReferenceLine
        x={final}
        stroke="var(--mantine-color-dark-3)"
        label={{
          value: 'This case',
          position: 'top',
          fontSize: 11,
          fill: 'var(--mantine-color-text)',
        }}
      />
      <Bar dataKey="range" isAnimationActive={false} radius={2}>
        {steps.map((s) => (
          <Cell key={s.key} fill={COLOR[s.kind]} />
        ))}
      </Bar>
    </BarChart>
  )

  return (
    <Stack gap={4}>
      <div style={{ width: '100%', height }} role="img" aria-label="SHAP waterfall chart">
        {fixedWidth ? (
          chart
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            {chart}
          </ResponsiveContainer>
        )}
      </div>
      <Group justify="space-between" px={8}>
        <Text size="xs" c="teal.7">
          ← lowers risk
        </Text>
        <Text size="xs" c="red.7">
          raises risk →
        </Text>
      </Group>
    </Stack>
  )
}
