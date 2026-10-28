import { AppShell, Badge, Burger, Button, Group, NavLink, Text, Title } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { NavLink as RouterNavLink, Outlet, useLocation } from 'react-router'

import { useAuth } from '../features/auth/context'
import { isAdmin } from '../features/auth/permissions'

type NavItem = {
  label: string
  to: string
  adminOnly?: boolean
  match: (path: string, search: string) => boolean
}

const NAV: NavItem[] = [
  {
    label: 'Review queue',
    to: '/queue',
    match: (path, search) => path === '/queue' && !search.includes('tab=mine'),
  },
  {
    label: 'My cases',
    to: '/queue?tab=mine',
    match: (path, search) => path === '/queue' && search.includes('tab=mine'),
  },
  {
    label: 'Monitoring',
    to: '/monitoring',
    match: (path) => path === '/monitoring',
  },
  {
    label: 'Threshold and costs',
    to: '/admin/settings',
    adminOnly: true,
    match: (path) => path === '/admin/settings',
  },
]

/** Logged-in page frame: header (app name, user, logout) and side navigation. */
export function AppLayout() {
  const auth = useAuth()
  const location = useLocation()
  const [opened, { toggle, close }] = useDisclosure()

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 220, breakpoint: 'sm', collapsed: { mobile: !opened } }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between" wrap="nowrap">
          <Group gap="sm" wrap="nowrap">
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <Title order={4}>Fraud review</Title>
          </Group>
          {auth.user && (
            <Group gap="sm" wrap="nowrap">
              <Text size="sm" visibleFrom="sm">
                {auth.user.full_name}
              </Text>
              <Badge variant="light" color={auth.user.role === 'admin' ? 'grape' : 'blue'}>
                {auth.user.role}
              </Badge>
              <Button variant="default" size="xs" onClick={() => void auth.logout()}>
                Log out
              </Button>
            </Group>
          )}
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="xs">
        {NAV.filter((item) => !item.adminOnly || isAdmin(auth.user)).map((item) => (
          <NavLink
            key={item.label}
            component={RouterNavLink}
            to={item.to}
            label={item.label}
            active={item.match(location.pathname, location.search)}
            onClick={close}
          />
        ))}
      </AppShell.Navbar>

      <AppShell.Main>
        <Outlet />
      </AppShell.Main>
    </AppShell>
  )
}
