import { useQuery } from '@tanstack/react-query'
import { apiClient } from './api'

export interface DiscoveredModel {
  id: string
  name: string
  family: string
  gated: boolean | string
  private: boolean
  downloads: number | null
  tags: string[]
  source: string
}

export interface ModelDiscoveryResult {
  models: DiscoveredModel[]
  source: string
  warning?: string | null
}

export interface HuggingFaceStatus { token_configured: boolean; message: string }

export function useModelDiscovery(query: string, enabled = true) {
  return useQuery<ModelDiscoveryResult>({
    queryKey: ['model-discovery', query],
    queryFn: () => apiClient.discoverModels(query),
    enabled,
    staleTime: 60_000,
    retry: 1,
  })
}

export function useHuggingFaceStatus() {
  return useQuery<HuggingFaceStatus>({
    queryKey: ['huggingface-status'],
    queryFn: () => apiClient.getHuggingFaceStatus(),
    staleTime: 30_000,
    retry: 1,
  })
}
