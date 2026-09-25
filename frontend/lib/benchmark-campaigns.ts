import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './api'

export interface BenchmarkCampaignRun {
  id: number
  model: string
  benchmark: string
  status: string
  score: number | null
  num_correct: number | null
  num_samples: number | null
  requested_samples: number
  truncated_count?: number | null
  invalid_answer_count?: number | null
  progress: Record<string, unknown> | null
  error: string | null
  provenance: Record<string, unknown> | null
  generation?: Record<string, unknown> | null
  scoring?: Record<string, unknown> | string | null
  model_provenance?: Record<string, unknown> | null
  rescoring?: { source_test_run_id: number; source_result_id: number } | null
}

export interface BenchmarkCampaign {
  id: string
  name: string
  created_at: string
  status: 'pending' | 'running' | 'completed' | 'partial' | 'failed' | 'cancelled'
  models: string[]
  benchmarks: string[]
  num_samples: number
  seed: number
  max_new_tokens: number
  completed: number
  total: number
  runs: BenchmarkCampaignRun[]
  comparable: boolean
  warnings: string[]
  scoring_outdated?: boolean
  source_campaign_id?: string
}

export interface BenchmarkCampaignRequest {
  name: string
  models: string[]
  benchmarks: string[]
  num_samples: number
  seed: number
  max_new_tokens: number
}

export const BENCHMARK_CHECKS = [
  { id: 'mmlu', name: 'MMLU', description: 'Knowledge across academic subjects', format: 'Multiple choice' },
  { id: 'gsm8k', name: 'GSM8K', description: 'Grade school math and reasoning', format: 'Numeric final answer' },
  { id: 'hellaswag', name: 'HellaSwag', description: 'Everyday reasoning and plausible continuations', format: 'Multiple choice' },
  { id: 'arc', name: 'ARC Challenge', description: 'Science questions with challenging distractors', format: 'Multiple choice' },
] as const

export const benchmarkName = (id: string) => BENCHMARK_CHECKS.find((check) => check.id === id)?.name ?? id
export const activeCampaign = (campaign: BenchmarkCampaign) => campaign.status === 'pending' || campaign.status === 'running'
export const shortModelName = (model: string) => model.split('/').filter(Boolean).pop() || model

export function useBenchmarkCampaigns() {
  return useQuery({
    queryKey: ['benchmark-campaigns'],
    queryFn: () => apiClient.listBenchmarkCampaigns(),
    refetchInterval: (query) => query.state.data?.campaigns.some(activeCampaign) ? 3000 : false,
  })
}

export function useBenchmarkCampaign(id: string | undefined) {
  return useQuery({
    queryKey: ['benchmark-campaign', id],
    queryFn: () => apiClient.getBenchmarkCampaign(id!),
    enabled: !!id,
    refetchInterval: (query) => query.state.data && activeCampaign(query.state.data) ? 3000 : false,
  })
}

export function useStartBenchmarkCampaign() {
  const cache = useQueryClient()
  return useMutation({
    mutationFn: (request: BenchmarkCampaignRequest) => apiClient.createBenchmarkCampaign(request),
    onSuccess: (campaign) => {
      cache.setQueryData(['benchmark-campaign', campaign.id], campaign)
      cache.invalidateQueries({ queryKey: ['benchmark-campaigns'] })
      cache.invalidateQueries({ queryKey: ['test-runs'] })
      cache.invalidateQueries({ queryKey: ['model-lineage'] })
    },
  })
}

export function useCancelBenchmarkCampaign() {
  const cache = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiClient.cancelBenchmarkCampaign(id),
    onSuccess: (campaign) => {
      cache.setQueryData(['benchmark-campaign', campaign.id], campaign)
      cache.invalidateQueries({ queryKey: ['benchmark-campaigns'] })
      cache.invalidateQueries({ queryKey: ['test-runs'] })
    },
  })
}

export function useRescoreBenchmarkCampaign() {
  const cache = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiClient.rescoreBenchmarkCampaign(id),
    onSuccess: (campaign) => {
      cache.setQueryData(['benchmark-campaign', campaign.id], campaign)
      cache.invalidateQueries({ queryKey: ['benchmark-campaigns'] })
      cache.invalidateQueries({ queryKey: ['test-runs'] })
      cache.invalidateQueries({ queryKey: ['model-lineage'] })
    },
  })
}
