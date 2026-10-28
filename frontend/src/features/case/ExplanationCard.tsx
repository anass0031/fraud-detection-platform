import { Badge, Divider, Group, List, Paper, Stack, Text, ThemeIcon, Title } from '@mantine/core'

import type { CaseDetail } from './api'
import { buildWaterfall, type Factor, orderFactors } from './waterfall'
import { WaterfallChart } from './WaterfallChart'

const STRENGTH_COLOR: Record<string, string> = {
  strongly: 'dark',
  moderately: 'gray',
  slightly: 'gray',
}

function FactorItem({ f }: { f: Factor }) {
  const raises = f.contribution >= 0
  return (
    <List.Item
      icon={
        <ThemeIcon size={20} radius="xl" variant="light" color={raises ? 'red' : 'teal'}>
          <Text span fz={10} aria-hidden>
            {raises ? '▲' : '▼'}
          </Text>
        </ThemeIcon>
      }
    >
      <Group gap={6} wrap="nowrap" align="flex-start">
        <Text size="sm">{f.text}</Text>
        {f.strength === 'strongly' && (
          <Badge size="xs" variant="outline" color={STRENGTH_COLOR[f.strength]}>
            strong
          </Badge>
        )}
      </Group>
    </List.Item>
  )
}

/** Why the model scored this case: SHAP waterfall + plain-language reasons. */
export function ExplanationCard({ c }: { c: CaseDetail }) {
  const p = c.prediction
  const waterfall = buildWaterfall(p)
  const { readable, masked } = orderFactors(p.top_factors)

  return (
    <Paper withBorder p="md">
      <Stack gap="sm">
        <div>
          <Title order={4}>Why this score</Title>
          <Text size="xs" c="dimmed">
            Starting from a typical transaction, each bar shows how much one factor pushed this
            case's risk up (red) or down (green). Hover a bar for details.
          </Text>
        </div>

        {p.top_factors.length === 0 ? (
          <Text size="sm" c="dimmed">
            No explanation stored for this prediction.
          </Text>
        ) : (
          <>
            <WaterfallChart waterfall={waterfall} />
            {!waterfall.complete && (
              <Text size="xs" c="dimmed">
                Partial explanation: this prediction was stored without its final value.
              </Text>
            )}

            <Divider label="Risk factors" labelPosition="left" />
            <List spacing={6} aria-label="Risk factors">
              {readable.map((f) => (
                <FactorItem key={f.key} f={f} />
              ))}
            </List>
            {masked.length > 0 && (
              <>
                <Text size="xs" c="dimmed" mt="xs">
                  Anonymised signals (the data provider hides what these columns mean):
                </Text>
                <List spacing={6} aria-label="Anonymised signals">
                  {masked.map((f) => (
                    <FactorItem key={f.key} f={f} />
                  ))}
                </List>
              </>
            )}
          </>
        )}
      </Stack>
    </Paper>
  )
}
