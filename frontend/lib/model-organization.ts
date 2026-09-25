import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import type { ModelLineage, ModelLineageNode } from '@/lib/model-catalog'

export interface ModelExperiment {
  id: string
  name: string
  notes: string
  created_at: string
}

export interface ModelOrganizationItem {
  label: string
  notes: string
  experiment_ids: string[]
}

export interface ModelOrganization {
  experiments: ModelExperiment[]
  items: Record<string, ModelOrganizationItem>
}

export const UNASSIGNED_EXPERIMENT = '__unassigned__'
export const modelOrganizationKey = (modelRef: string) => `model:${modelRef}`
export const stepOrganizationKey = (node: ModelLineageNode) => node.kind === 'model' && node.model_ref ? modelOrganizationKey(node.model_ref) : `step:${node.id}`

export function belongsToExperiment(item: ModelOrganizationItem | undefined, experimentId: string) {
  if (!experimentId) return true
  return experimentId === UNASSIGNED_EXPERIMENT ? !item?.experiment_ids.length : !!item?.experiment_ids.includes(experimentId)
}

/** Keep real ancestors for context; grouping never changes recorded relationships. */
export function filterExperimentLineage(graph: Pick<ModelLineage, 'nodes' | 'edges'>, organization: ModelOrganization | undefined, experimentId: string) {
  const assigned = new Set(graph.nodes.filter((node) => belongsToExperiment(organization?.items[stepOrganizationKey(node)], experimentId)).map((node) => node.id))
  const included = new Set(assigned)
  const pending = Array.from(assigned)
  while (pending.length) {
    const id = pending.pop()!
    for (const edge of graph.edges) {
      if (edge.target === id && !included.has(edge.source)) { included.add(edge.source); pending.push(edge.source) }
    }
  }
  return {
    nodes: graph.nodes.filter((node) => included.has(node.id)),
    edges: graph.edges.filter((edge) => included.has(edge.source) && included.has(edge.target)),
    context: new Set(Array.from(included).filter((id) => !assigned.has(id))),
  }
}

export function useModelOrganization() {
  return useQuery<ModelOrganization>({ queryKey: ['model-organization'], queryFn: () => apiClient.getModelOrganization(), staleTime: 15_000 })
}

export function useSaveOrganizationItem() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: ModelOrganizationItem & { key: string }) => apiClient.saveModelOrganizationItem(data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['model-organization'] }),
  })
}

export function useSaveExperiment() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: { id?: string; name: string; notes: string }) => data.id ? apiClient.updateModelExperiment(data.id, { name: data.name, notes: data.notes }) : apiClient.createModelExperiment({ name: data.name, notes: data.notes }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['model-organization'] }),
  })
}
