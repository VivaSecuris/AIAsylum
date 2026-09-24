import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import {
  apiClient,
  InterpRunRequest,
  WeightRunRequest,
  TestRun,
  TestRunRequest,
  TestResult,
  ConversationTurn,
  Assessment,
  Prompt,
  PromptCreate,
  PromptUpdate,
  Suite,
  SuiteProgress,
} from './api'

// Test Runs
export function useTestRuns(params?: { limit?: number; offset?: number; test_type?: string }) {
  return useQuery({
    queryKey: ['test-runs', params],
    queryFn: () => apiClient.listTestRuns(params),
  })
}

export function useTestRun(id: number) {
  return useQuery({
    queryKey: ['test-run', id],
    queryFn: () => apiClient.getTestRun(id),
    enabled: !!id,
  })
}

export function useCreateTestRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: TestRunRequest) => apiClient.createTestRun(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
    },
  })
}

export function useStartTestRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.startTestRun(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['test-run', id] })
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
    },
  })
}

export function usePauseTestRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.pauseTestRun(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['test-run', id] })
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
    },
  })
}

export function useResumeTestRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.resumeTestRun(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['test-run', id] })
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
    },
  })
}

export function useStopTestRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.stopTestRun(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['test-run', id] })
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
    },
  })
}

export function useUpdateTestRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: { name?: string } }) =>
      apiClient.updateTestRun(id, data),
    onSuccess: (updatedRun, { id, data }) => {
      // Optimistically update the cache immediately
      queryClient.setQueryData(['test-run', id], updatedRun)
      
      // Update all test-runs list caches (with different params)
      queryClient.setQueriesData({ queryKey: ['test-runs'] }, (oldData: TestRun[] | undefined) => {
        if (!oldData) return oldData
        return oldData.map(run => {
          if (run.id === id) {
            // Update the run with new meta_data
            const newMetaData = { ...run.meta_data, name: data.name }
            // Remove name if it's empty string
            if (data.name === '') {
              delete newMetaData.name
            }
            return { ...run, meta_data: newMetaData }
          }
          return run
        })
      })
      
      // Invalidate to ensure fresh data from server
      queryClient.invalidateQueries({ queryKey: ['test-run', id] })
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
      queryClient.invalidateQueries({ queryKey: ['suite-runs'] })
    },
  })
}

export function useDeleteTestRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => {
      console.log('useDeleteTestRun: Calling API to delete test run', id)
      return apiClient.deleteTestRun(id)
    },
    onSuccess: (_, id) => {
      console.log('useDeleteTestRun: Delete successful, invalidating queries', id)
      // Invalidate all test-runs queries
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
      queryClient.invalidateQueries({ queryKey: ['test-run', id] })
      // Also remove from cache immediately
      queryClient.removeQueries({ queryKey: ['test-run', id] })
    },
    onError: (error: any) => {
      console.error('useDeleteTestRun: Delete failed', error)
    },
  })
}

export function useTestResults(testRunId: number) {
  return useQuery({
    queryKey: ['test-results', testRunId],
    queryFn: () => apiClient.getTestResults(testRunId),
    enabled: !!testRunId,
  })
}

export function useConversation(testRunId: number) {
  return useQuery({
    queryKey: ['conversation', testRunId],
    queryFn: async () => {
      const data = await apiClient.getConversation(testRunId)
      console.log('[useConversation] Fetched conversation data:', data?.length || 0, data)
      return data
    },
    enabled: !!testRunId,
    refetchOnWindowFocus: true,
    refetchInterval: false, // We'll handle refetching via SSE events
  })
}

// Analysis
export function useAssessments(testRunId: number) {
  return useQuery({
    queryKey: ['assessments', testRunId],
    queryFn: () => apiClient.getAssessments(testRunId),
    enabled: !!testRunId,
  })
}

export function useMultipleAssessments(testRunIds: number[]) {
  return useQuery({
    queryKey: ['assessments', 'multiple', testRunIds],
    queryFn: async () => {
      // Fetch assessments for all test runs in parallel
      const results = await Promise.all(
        testRunIds.map(id => 
          apiClient.getAssessments(id).catch(err => {
            console.warn(`Failed to fetch assessments for test run ${id}:`, err)
            return []
          })
        )
      )
      // Return a map of test_run_id -> assessments[]
      const map = new Map<number, Assessment[]>()
      testRunIds.forEach((id, index) => {
        map.set(id, results[index] || [])
      })
      return map
    },
    enabled: testRunIds.length > 0,
  })
}

export function useRunAnalysis() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ testRunId, config }: { testRunId: number; config?: Record<string, any> }) =>
      apiClient.runAnalysis(testRunId, config),
    onSuccess: (_, { testRunId }) => {
      queryClient.invalidateQueries({ queryKey: ['assessments', testRunId] })
      queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
    },
  })
}

export function useRunAnalysisUnanalyzed() {
  const queryClient = useQueryClient()
  // `| void` lets callers run with the default config via mutateAsync()
  return useMutation({
    mutationFn: (config: Record<string, any> | void) =>
      apiClient.runAnalysisUnanalyzed(config || undefined),
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
      queryClient.invalidateQueries({ queryKey: ['assessments'] })
      await queryClient.refetchQueries({ queryKey: ['test-runs'] })
    },
  })
}

// Benchmarks
export function useBenchmarks() {
  return useQuery({
    queryKey: ['benchmarks'],
    queryFn: () => apiClient.listBenchmarks(),
  })
}

export function useRunBenchmark() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: { provider: string; model: string; benchmark: string; num_samples?: number }) =>
      apiClient.runBenchmark(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
    },
  })
}

// Prompts
export function usePrompts(params?: Parameters<typeof apiClient.listPrompts>[0]) {
  return useQuery({
    queryKey: ['prompts', params],
    queryFn: () => apiClient.listPrompts(params),
  })
}

export function usePrompt(id: number) {
  return useQuery({
    queryKey: ['prompt', id],
    queryFn: () => apiClient.getPrompt(id),
    enabled: !!id,
  })
}

export function useCreatePrompt() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: PromptCreate) => apiClient.createPrompt(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompts'] })
    },
  })
}

export function useUpdatePrompt() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: PromptUpdate }) => apiClient.updatePrompt(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['prompts'] })
      queryClient.invalidateQueries({ queryKey: ['prompt', id] })
    },
  })
}

export function useDeletePrompt() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.deletePrompt(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompts'] })
    },
  })
}

export function usePromptVariables(promptId: number | undefined) {
  return useQuery({
    queryKey: ['prompt-variables', promptId],
    queryFn: () => apiClient.getPromptVariables(promptId!),
    enabled: !!promptId,
  })
}

// Suites
export function useSuites(params?: { limit?: number; offset?: number }) {
  return useQuery({
    queryKey: ['suites', params],
    queryFn: () => apiClient.listSuites(params),
  })
}

export function useSuite(id: number) {
  return useQuery({
    queryKey: ['suite', id],
    queryFn: () => apiClient.getSuite(id),
    enabled: !!id,
  })
}

export function useCreateSuite() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: {
      name?: string
      test_types: string[]
      benchmarks: string[]
      models: Array<{ provider: string; model: string }>
      test_config?: Record<string, any>
      num_samples?: number
    }) => apiClient.createSuite(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['suites'] })
    },
  })
}

export function useUpdateSuite() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: { name?: string } }) =>
      apiClient.updateSuite(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['suite', id] })
      queryClient.invalidateQueries({ queryKey: ['suites'] })
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
    },
  })
}

export function useSuiteRuns(suiteId: number) {
  return useQuery({
    queryKey: ['suite-runs', suiteId],
    queryFn: () => apiClient.getSuiteRuns(suiteId),
    enabled: !!suiteId,
  })
}

export function useSuiteProgress(suiteId: number, enabled: boolean = true) {
  return useQuery({
    queryKey: ['suite-progress', suiteId],
    queryFn: () => apiClient.getSuiteProgress(suiteId),
    enabled: !!suiteId && enabled,
    refetchInterval: (query) => {
      // Auto-refresh every 2-3 seconds if suite is running (react-query v5 passes the Query, not its data)
      const data = query.state.data
      if (data?.running_runs && data.running_runs > 0) {
        return 2000
      }
      return false
    },
  })
}

export function useDeleteSuite() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.deleteSuite(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['suites'] })
    },
  })
}

// Progress monitoring with SSE
export interface ProgressEvent {
  test_run_id: number
  event_type: string
  timestamp: string
  data?: Record<string, any>
  message?: string
}

export function useTestRunProgress(testRunId: number, enabled: boolean = true) {
  const queryClient = useQueryClient()
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!enabled || !testRunId || testRunId === 0) {
      console.log('[useTestRunProgress] Disabled:', { enabled, testRunId })
      return
    }

    console.log('[useTestRunProgress] Connecting to SSE for test run', testRunId)
    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const url = `${API_BASE_URL}/api/v1/test-runs/${testRunId}/progress`
    console.log('[useTestRunProgress] SSE URL:', url)
    
    const eventSource = new EventSource(url)

    eventSource.onopen = () => {
      console.log('[useTestRunProgress] SSE connection opened')
      setIsConnected(true)
      setError(null)
    }

    eventSource.onmessage = (event) => {
      try {
        console.log('[useTestRunProgress] Received event:', event.data)
        const data: ProgressEvent = JSON.parse(event.data)
        console.log('[useTestRunProgress] Parsed event:', data)
        setProgress(data)

        // Invalidate queries based on event type
        if (data.event_type === 'test_completed' || data.event_type === 'test_failed' || data.event_type === 'test_cancelled') {
          console.log('[useTestRunProgress] Test completed/failed/cancelled, invalidating queries')
          queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
          queryClient.invalidateQueries({ queryKey: ['test-results', testRunId] })
          queryClient.invalidateQueries({ queryKey: ['conversation', testRunId] })
        } else if (data.event_type === 'benchmark_progress' || data.event_type === 'test_started') {
          queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
        } else if (data.event_type === 'conversation_turn') {
          // Immediately refresh conversation when a new turn is saved
          console.log('[useTestRunProgress] New conversation turn, refreshing conversation')
          // Invalidate and immediately refetch to get the latest conversation
          queryClient.invalidateQueries({ queryKey: ['conversation', testRunId] })
          queryClient.refetchQueries({ queryKey: ['conversation', testRunId] })
        }
      } catch (error) {
        console.error('[useTestRunProgress] Error parsing progress event:', error, event.data)
        setError(`Parse error: ${error}`)
      }
    }

    eventSource.onerror = (error) => {
      console.error('[useTestRunProgress] SSE connection error:', error)
      console.error('[useTestRunProgress] EventSource readyState:', eventSource.readyState)
      setIsConnected(false)
      if (eventSource.readyState === EventSource.CLOSED) {
        setError('Connection closed')
        eventSource.close()
      } else if (eventSource.readyState === EventSource.CONNECTING) {
        setError('Connecting...')
      }
    }

    return () => {
      console.log('[useTestRunProgress] Cleaning up SSE connection')
      eventSource.close()
      setIsConnected(false)
    }
  }, [testRunId, enabled, queryClient])

  return { progress, isConnected, error }
}

// ---------------------------------------------------------------- Interpretability

export function useInterpModes() {
  return useQuery({
    queryKey: ['interp-modes'],
    queryFn: () => apiClient.listInterpModes(),
    // Includes live memory warnings, so it should not be cached for long.
    staleTime: 15_000,
  })
}

export function useInterpRuns(params?: { limit?: number; mode?: string }) {
  return useQuery({
    queryKey: ['interp-runs', params],
    queryFn: () => apiClient.listInterpRuns(params),
  })
}

export function useInterpRun(id: number) {
  return useQuery({
    queryKey: ['interp-run', id],
    queryFn: () => apiClient.getInterpRun(id),
    enabled: !!id && id > 0,
  })
}

export function useCreateInterpRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: InterpRunRequest) => apiClient.createInterpRun(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['interp-runs'] })
    },
  })
}

export function useDeleteInterpRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.deleteInterpRun(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['interp-runs'] })
    },
  })
}

export function useStopInterpRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.stopInterpRun(id),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: ['interp-run', id] })
      queryClient.invalidateQueries({ queryKey: ['interp-runs'] })
    },
  })
}

/**
 * Live progress for an interpretability run.
 *
 * Mirrors useTestRunProgress but against the interp stream. The two are
 * deliberately separate endpoints: the backend keeps its own event manager for
 * interp runs so that interp run 5 and test run 5 are not the same
 * subscription.
 */
export function useInterpRunProgress(runId: number, enabled: boolean = true) {
  const queryClient = useQueryClient()
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!enabled || !runId || runId === 0) return

    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const eventSource = new EventSource(`${API_BASE_URL}/api/v1/interp/runs/${runId}/progress`)

    eventSource.onopen = () => {
      setIsConnected(true)
      setError(null)
    }

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data)
        setProgress(data)
        if (
          data.event_type === 'interp_completed' ||
          data.event_type === 'interp_failed' ||
          data.event_type === 'interp_cancelled'
        ) {
          queryClient.invalidateQueries({ queryKey: ['interp-run', runId] })
          queryClient.invalidateQueries({ queryKey: ['interp-runs'] })
        }
      } catch (err) {
        setError(`Parse error: ${err}`)
      }
    }

    eventSource.onerror = () => {
      setIsConnected(false)
      if (eventSource.readyState === EventSource.CLOSED) {
        setError('Connection closed')
        eventSource.close()
      }
    }

    return () => {
      eventSource.close()
      setIsConnected(false)
    }
  }, [runId, enabled, queryClient])

  return { progress, isConnected, error }
}


// ---------------------------------------------------------------- Weight surgery

export function useWeightStages() {
  return useQuery({
    queryKey: ['weight-stages'],
    queryFn: () => apiClient.listWeightStages(),
    // Carries the live model-slot status, so it should not be cached for long.
    staleTime: 15_000,
  })
}

export function useWeightObjectives() {
  return useQuery({
    queryKey: ['weight-objectives'],
    queryFn: () => apiClient.listWeightObjectives(),
    staleTime: 60_000,
  })
}

/**
 * Machine-state preflight for a prospective run.
 *
 * Kept short-lived rather than cached: the conditions it reports -- a resident
 * Ollama model, swap pressure, free disk -- change while the form is open, and
 * a stale "all clear" is worse than none at all.
 */
export function useWeightPreflight(
  params: { kind: string; source_model?: string; source_run_id?: number; output_name?: string },
  enabled = true,
) {
  return useQuery({
    queryKey: ['weight-preflight', params],
    queryFn: () => apiClient.weightPreflight(params),
    enabled: enabled && !!params.kind,
    staleTime: 5_000,
  })
}

export function useWeightRuns(params?: { limit?: number; kind?: string; status?: string }) {
  return useQuery({
    queryKey: ['weight-runs', params],
    queryFn: () => apiClient.listWeightRuns(params),
  })
}

export function useWeightRun(id: number, isActive = false) {
  return useQuery({
    queryKey: ['weight-run', id],
    queryFn: () => apiClient.getWeightRun(id),
    enabled: !!id && id > 0,
    // Insurance behind the SSE stream. A surgery job's steps are minutes
    // apart, so a dropped stream would otherwise leave the page frozen and
    // indistinguishable from a hang.
    refetchInterval: isActive ? 10_000 : false,
  })
}

export function useDirections(usableOnly = false) {
  return useQuery({
    queryKey: ['weight-directions', usableOnly],
    queryFn: () => apiClient.listDirections(usableOnly),
  })
}

export function useEditedModels() {
  return useQuery({
    queryKey: ['weight-models'],
    queryFn: () => apiClient.listEditedModels(),
  })
}

export function useEditedModel(name: string) {
  return useQuery({
    queryKey: ['weight-model', name],
    queryFn: () => apiClient.getEditedModel(name),
    enabled: !!name,
  })
}

export function useCreateWeightRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: WeightRunRequest) => apiClient.createWeightRun(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['weight-runs'] })
      queryClient.invalidateQueries({ queryKey: ['weight-directions'] })
    },
  })
}

export function useDeleteWeightRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (args: {
      id: number
      delete_artifacts?: boolean
      confirm?: string
      force?: boolean
    }) => apiClient.deleteWeightRun(args.id, args),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['weight-runs'] })
      queryClient.invalidateQueries({ queryKey: ['weight-directions'] })
      queryClient.invalidateQueries({ queryKey: ['weight-models'] })
    },
  })
}

export function useStopWeightRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiClient.stopWeightRun(id),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: ['weight-run', id] })
      queryClient.invalidateQueries({ queryKey: ['weight-runs'] })
    },
  })
}

/**
 * Live progress for a weight run.
 *
 * A third stream alongside test runs and interp runs, because all three
 * backends key their event managers on bare integers -- so weight run 5 and
 * interp run 5 must not become the same subscription.
 */
export function useWeightRunProgress(runId: number, enabled: boolean = true) {
  const queryClient = useQueryClient()
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!enabled || !runId || runId === 0) return

    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const eventSource = new EventSource(`${API_BASE_URL}/api/v1/weights/runs/${runId}/progress`)

    eventSource.onopen = () => {
      setIsConnected(true)
      setError(null)
    }

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressEvent = JSON.parse(event.data)
        setProgress(data)
        if (
          data.event_type === 'weights_completed' ||
          data.event_type === 'weights_failed' ||
          data.event_type === 'weights_cancelled'
        ) {
          queryClient.invalidateQueries({ queryKey: ['weight-run', runId] })
          queryClient.invalidateQueries({ queryKey: ['weight-runs'] })
          queryClient.invalidateQueries({ queryKey: ['weight-directions'] })
          queryClient.invalidateQueries({ queryKey: ['weight-models'] })
        }
      } catch (err) {
        setError(`Parse error: ${err}`)
      }
    }

    eventSource.onerror = () => {
      setIsConnected(false)
      if (eventSource.readyState === EventSource.CLOSED) {
        setError('Connection closed')
        eventSource.close()
      }
    }

    return () => {
      eventSource.close()
      setIsConnected(false)
    }
  }, [runId, enabled, queryClient])

  return { progress, isConnected, error }
}

// ---------------------------------------------------------------- Interp artifacts

export function useInterpArtifacts(runId: number, enabled = true) {
  return useQuery({
    queryKey: ['interp-artifacts', runId],
    queryFn: () => apiClient.listInterpArtifacts(runId),
    enabled: enabled && !!runId && runId > 0,
  })
}

export function useInterpArtifact(runId: number, name: string, enabled = true) {
  return useQuery({
    queryKey: ['interp-artifact', runId, name],
    queryFn: () => apiClient.getInterpArtifact(runId, name),
    enabled: enabled && !!runId && runId > 0 && !!name,
    // Payloads are written once when the run completes and never change.
    staleTime: Infinity,
    retry: false,
  })
}

export function useChatWithModel(name: string) {
  return useMutation({
    mutationFn: (data: {
      messages: Array<{ role: string; content: string }>
      system_prompt?: string
      temperature?: number
      max_tokens?: number
    }) => apiClient.chatWithModel(name, data),
  })
}
