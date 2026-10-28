import { Button, Center, Group, Stack, Text, Title } from '@mantine/core'
import { Link, useRouteError } from 'react-router'

function ErrorPage({ code, message }: { code: string; message: string }) {
  return (
    <Center mih="60vh">
      <Stack align="center" gap="xs">
        <Title order={1} c="dimmed">
          {code}
        </Title>
        <Text>{message}</Text>
        <Button component={Link} to="/queue" variant="light" mt="sm">
          Back to the review queue
        </Button>
      </Stack>
    </Center>
  )
}

export function ForbiddenPage() {
  return <ErrorPage code="403" message="You do not have access to this page." />
}

export function NotFoundPage() {
  return <ErrorPage code="404" message="This page does not exist." />
}

/** Shown when a page crashes or its code cannot be downloaded (e.g. after a new deploy). */
export function RouteErrorPage() {
  const error = useRouteError()
  const message = error instanceof Error ? error.message : 'Unexpected error'
  return (
    <Center mih="60vh" p="md">
      <Stack align="center" gap="xs" maw={480}>
        <Title order={2}>Something went wrong</Title>
        <Text c="dimmed" ta="center" size="sm">
          {message}
        </Text>
        <Group mt="sm">
          <Button onClick={() => window.location.reload()}>Reload the page</Button>
          <Button component={Link} to="/queue" variant="default" reloadDocument>
            Review queue
          </Button>
        </Group>
      </Stack>
    </Center>
  )
}
