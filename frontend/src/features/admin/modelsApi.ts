import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { type Schemas, api, errorMessage, unwrap } from '../../api/client'

export type ModelVersion = Schemas['ModelOut']

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: async () => unwrap(await api.GET('/api/v1/models')),
  })
}

/** After a change: the registry, and everything that follows the active model. */
function useRefresh() {
  const queryClient = useQueryClient()
  return () =>
    Promise.all(
      ['models', 'monitoring', 'cases', 'case'].map((key) =>
        queryClient.invalidateQueries({ queryKey: [key] }),
      ),
    )
}

export function useActivateModel() {
  const refresh = useRefresh()
  return useMutation({
    mutationFn: async (id: number) =>
      unwrap(
        await api.POST('/api/v1/models/{model_id}/activate', {
          params: { path: { model_id: id } },
        }),
      ),
    onSuccess: (res) => {
      notifications.show({
        color: 'green',
        title: `${res.model.version} is now active`,
        message: res.threshold_override_reset
          ? 'The custom alert threshold was reset to this model’s own threshold.'
          : 'New transactions are scored with it from now on.',
      })
      return refresh()
    },
    onError: (error) => {
      notifications.show({ color: 'red', title: 'Not activated', message: errorMessage(error) })
      return refresh()
    },
  })
}

export function useArchiveModel() {
  const refresh = useRefresh()
  return useMutation({
    mutationFn: async (id: number) =>
      unwrap(
        await api.POST('/api/v1/models/{model_id}/archive', { params: { path: { model_id: id } } }),
      ),
    onSuccess: (m) => {
      notifications.show({ color: 'gray', message: `${m.version} archived` })
      return refresh()
    },
    onError: (error) => {
      notifications.show({ color: 'red', title: 'Not archived', message: errorMessage(error) })
      return refresh()
    },
  })
}
