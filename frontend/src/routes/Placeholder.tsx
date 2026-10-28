import { Text, Title } from '@mantine/core'

/** Temporary page body while a page is being built. */
export function Placeholder({ title }: { title: string }) {
  return (
    <>
      <Title order={2}>{title}</Title>
      <Text c="dimmed">Coming next.</Text>
    </>
  )
}
