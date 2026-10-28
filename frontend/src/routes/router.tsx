import { Center, Loader } from '@mantine/core'
import { Navigate, type RouteObject, createBrowserRouter } from 'react-router'

import { LoginPage } from '../features/auth/LoginPage'
import { AppLayout } from '../layouts/AppLayout'
import { ForbiddenPage, NotFoundPage, RouteErrorPage } from './ErrorPages'
import { RequireAuth, RequireRole } from './guards'

/** All routes. Everything except the login page needs a logged-in user. */
export const routes: RouteObject[] = [
  { path: '/login', element: <LoginPage />, errorElement: <RouteErrorPage /> },
  {
    path: '/',
    element: (
      <RequireAuth>
        <AppLayout />
      </RequireAuth>
    ),
    errorElement: <RouteErrorPage />,
    // Shown while a lazily loaded page's code downloads on a direct visit or reload.
    hydrateFallbackElement: (
      <Center h="100vh">
        <Loader aria-label="Loading page" />
      </Center>
    ),
    children: [
      { index: true, element: <Navigate to="/queue" replace /> },
      {
        path: 'queue',
        // Loaded after login: the table, filters and date picker are not needed before.
        lazy: async () => ({ Component: (await import('../features/queue/QueuePage')).QueuePage }),
      },
      {
        path: 'cases/:caseId',
        // Loaded on first use: the case page carries the charting library (Recharts).
        lazy: async () => ({ Component: (await import('../features/case/CasePage')).CasePage }),
      },
      {
        path: 'monitoring',
        // Loaded on first use: charts.
        lazy: async () => ({
          Component: (await import('../features/monitoring/MonitoringPage')).MonitoringPage,
        }),
      },
      {
        path: 'admin',
        element: <RequireRole roles={['admin']} />,
        children: [
          {
            path: 'settings',
            lazy: async () => ({
              Component: (await import('../features/admin/SettingsPage')).SettingsPage,
            }),
          },
          {
            path: 'models',
            lazy: async () => ({
              Component: (await import('../features/admin/ModelsPage')).ModelsPage,
            }),
          },
        ],
      },
      { path: '403', element: <ForbiddenPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
]

export const router = createBrowserRouter(routes)
