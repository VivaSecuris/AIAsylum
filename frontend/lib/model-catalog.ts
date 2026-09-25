import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import { useEffect } from 'react'

export type ModelAvailability = 'ready' | 'missing' | 'incomplete' | 'download_required' | 'deleted'

export interface ModelCatalogEntry {
  id: string
  model_ref: string
  name: string
  kind: 'base' | 'custom'
  gated?: boolean | string
  family?: string
  provider: string
  source_model: string | null
  availability: ModelAvailability
  reason: string | null
  size_bytes: number | null
  created_at: string | null
  run_id: number | null
  manifest: Record<string, any> | null
  history: { test_runs: number; interp_runs: number; weight_runs: number }
}

export interface ModelCatalog {
  models: ModelCatalogEntry[]
  location: string
}

export interface ModelLineageNode {
  id: string
  label: string
  kind: 'model' | 'direction' | 'sweep' | 'select' | 'surgery' | 'expert_surgery' | 'routing' | 'lora' | 'distill' | 'compare' | 'probe' | 'interp' | 'test' | 'missing'
  status: string
  created_at: string | null
  model_ref: string | null
  source_model: string | null
  run_id: number | null
  origin: string
  settings: Record<string, unknown>
  summary: Record<string, unknown>
  links: { run?: string; resume?: string; comparison?: string }
  missing_reason?: string
}

export interface ModelLineageEdge { source: string; target: string; label: string }

export interface ModelLineage {
  nodes: ModelLineageNode[]
  edges: ModelLineageEdge[]
  warnings: string[]
}

export function useModelLineage() {
  return useQuery<ModelLineage>({
    queryKey: ['model-lineage'],
    queryFn: () => apiClient.getModelLineage(),
    staleTime: 10_000,
  })
}

export function useModelCatalog() {
  return useQuery<ModelCatalog>({
    queryKey: ['model-catalog'],
    queryFn: () => apiClient.listModelCatalog(),
    staleTime: 15_000,
  })
}

// Downloads into the server's Hugging Face cache. A download is a background
// job on the server; the list polls while any is active.
export type ModelDownloadStatus = 'queued' | 'resolving' | 'downloading' | 'completed' | 'failed' | 'cancelled'

export interface ModelDownload {
  id: string
  repo_id: string
  revision: string
  status: ModelDownloadStatus
  active: boolean
  bytes_done: number
  bytes_total: number | null
  files_total: number | null
  commit: string | null
  snapshot: string | null
  error: string | null
  already_present: boolean
  created_at: number
  completed_at: number | null
  elapsed_seconds: number
  log_path: string | null
}

export interface ModelDownloads {
  downloads: ModelDownload[]
  cache_dir: string
  free_bytes: number | null
  total_bytes: number | null
}

export function useModelDownloads() {
  const queryClient = useQueryClient()
  const query = useQuery<ModelDownloads>({
    queryKey: ['model-downloads'],
    queryFn: () => apiClient.listModelDownloads(),
    refetchInterval: (query) => (query.state.data?.downloads.some((d) => d.active) ? 2000 : false),
  })
  const completed = query.data?.downloads.filter((item) => item.status === 'completed').map((item) => item.id).sort().join(',') ?? ''
  useEffect(() => {
    if (completed) {
      queryClient.invalidateQueries({ queryKey: ['model-catalog'] })
      queryClient.invalidateQueries({ queryKey: ['model-lineage'] })
    }
  }, [completed, queryClient])
  return query
}

function useDownloadMutation<TArg, TResult>(fn: (arg: TArg) => Promise<TResult>) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['model-downloads'] })
      queryClient.invalidateQueries({ queryKey: ['model-catalog'] })
    },
  })
}

export function useStartModelDownload() {
  return useDownloadMutation((data: { repo_id: string; revision?: string }) => apiClient.startModelDownload(data))
}

export function useCancelModelDownload() {
  return useDownloadMutation((id: string) => apiClient.cancelModelDownload(id))
}

export function useDeleteCachedModel() {
  return useDownloadMutation((repoId: string) => apiClient.deleteCachedModel(repoId))
}

export function useDeleteCustomModels() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (names: string[]) => apiClient.deleteCustomModels(names),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ['model-catalog'] })
      client.invalidateQueries({ queryKey: ['model-lineage'] })
      client.invalidateQueries({ queryKey: ['weight-models'] })
    },
  })
}

/** A cached base model is addressed by its Hub id; a local path is not in the cache. */
export function isCachedBaseModel(model: ModelCatalogEntry) {
  return model.kind === 'base' && model.availability === 'ready' && !model.model_ref.startsWith('/')
}

export function canUseModel(model: ModelCatalogEntry) {
  return model.availability === 'ready' || (model.kind === 'base' && model.availability === 'download_required')
}

export function availabilityLabel(model: ModelCatalogEntry) {
  return {
    ready: 'Ready on server',
    missing: 'Not on this server',
    incomplete: 'Incomplete checkpoint',
    download_required: 'Download on first run',
    deleted: 'Deleted · history retained',
  }[model.availability]
}

export function editDescription(model: ModelCatalogEntry) {
  const manifest = model.manifest
  if (!manifest) return 'Saved custom checkpoint'
  const method = manifest.method || manifest.extra?.method
  const extra = manifest.extra ?? {}
  const parts = [method ? String(method).replace(/_/g, ' ') : 'Edited checkpoint']
  if (method === 'lora_merge') {
    if (extra.distill?.teacher_model) parts.push(`distilled from ${String(extra.distill.teacher_model).split('/').pop()} (${extra.distill.level ?? 'response'})`)
    if (extra.lora?.rank != null) parts.push(`rank ${extra.lora.rank}`)
    if (extra.train?.steps != null) parts.push(`${extra.train.steps} steps`)
  }
  if (manifest.coverage_verified === false) parts.push(`partial: ${extra.experts_edited ?? '?'} experts in ${extra.layers_edited ?? '?'} layers`)
  if (extra.expert_mode === 'ablate' && extra.expert_scale != null) parts.push(`scale ${extra.expert_scale}`)
  if (manifest.beta != null) parts.push(`β ${manifest.beta}`)
  if (extra.k != null) parts.push(`strength ${extra.k}`)
  if (extra.objective) parts.push(String(extra.objective).replace(/_/g, ' '))
  return parts.join(' · ')
}
