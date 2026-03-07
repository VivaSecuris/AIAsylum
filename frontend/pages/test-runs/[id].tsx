import { useRouter } from 'next/router'
import Link from 'next/link'
import { useState, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { ConversationViewer } from '@/components/conversation/ConversationViewer'
import { DataTable } from '@/components/common/DataTable'
import { AnalysisConfigDialog, AnalysisConfig } from '@/components/analysis/AnalysisConfigDialog'
import { useTestRun, useConversation, useTestResults, useAssessments, useRunAnalysis, useStartTestRun, usePauseTestRun, useResumeTestRun, useStopTestRun, useDeleteTestRun, useTestRunProgress } from '@/lib/hooks'
import { formatDate, formatDateTime } from '@/lib/utils'
import { TestResult, Assessment } from '@/lib/api'
import { Play, Download, GitCompare, Trash2, Square, Pause, RotateCw } from 'lucide-react'
import * as Tabs from '@radix-ui/react-tabs'
import { toast } from '@/lib/toast'

export default function TestRunDetailPage() {
  const router = useRouter()
  const { id } = router.query
  const testRunId = typeof id === 'string' ? parseInt(id) : 0

  const queryClient = useQueryClient()
  const { data: testRun, isLoading: loadingRun, error: runError } = useTestRun(testRunId)
  // For analysis runs, conversation/results/assessments belong to the source run, not the analysis run itself
  const dataRunId: number = (testRun?.test_type === 'analysis' && testRun?.meta_data?.source_test_run_id)
    ? testRun.meta_data.source_test_run_id
    : testRunId
  const { data: conversation = [], isLoading: loadingConv, error: convError } = useConversation(dataRunId)
  const { data: results = [], isLoading: loadingResults, error: resultsError } = useTestResults(dataRunId)
  const { data: assessments = [], isLoading: loadingAssessments } = useAssessments(dataRunId)
  const runAnalysis = useRunAnalysis()
  const startTestRun = useStartTestRun()
  const pauseTestRun = usePauseTestRun()
  const resumeTestRun = useResumeTestRun()
  const stopTestRun = useStopTestRun()
  const deleteTestRun = useDeleteTestRun()
  
  // State declarations (must be before useEffect hooks that use them)
  const [activeTab, setActiveTab] = useState('overview')
  const [showAnalysisDialog, setShowAnalysisDialog] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState<{step?: string; message?: string} | null>(null)
  
  // Use SSE for live progress monitoring when test is running, pending, or when analysis is running
  // Enable if testRun is loading (undefined) or if status is running/pending, OR if analysis is pending
  const shouldMonitor = !testRun || testRun?.status === 'running' || testRun?.status === 'pending' || runAnalysis.isPending
  const { progress, isConnected, error: progressError } = useTestRunProgress(
    testRunId, 
    shouldMonitor && testRunId > 0
  )
  
  // Debug logging
  useEffect(() => {
    if (testRunId > 0) {
      console.log('[TestRunDetailPage] Monitoring status:', {
        testRunId,
        shouldMonitor,
        testRunStatus: testRun?.status,
        isConnected,
        progressError,
        hasProgress: !!progress
      })
    }
  }, [testRunId, shouldMonitor, testRun?.status, isConnected, progressError, progress])
  
  // Auto-switch to Live tab when test starts running
  useEffect(() => {
    if (testRun?.status === 'running' && activeTab === 'overview') {
      setActiveTab('live')
    }
  }, [testRun?.status, activeTab])

  // When Analysis tab is hidden (test not completed and not an analysis run), switch to overview if we're on Analysis
  const showAnalysisTab = testRun?.status === 'completed' || testRun?.test_type === 'analysis'
  useEffect(() => {
    if (!showAnalysisTab && activeTab === 'analysis') {
      setActiveTab('overview')
    }
  }, [showAnalysisTab, activeTab])

  // Auto-scroll to bottom when new messages arrive in Live tab
  useEffect(() => {
    if (activeTab === 'live' && conversation.length > 0) {
      const container = document.getElementById('live-messages-container')
      if (container) {
        // Small delay to ensure DOM is updated
        setTimeout(() => {
          container.scrollTop = container.scrollHeight
        }, 100)
      }
    }
  }, [conversation, activeTab])

  // Aggressive polling for conversation and results when test is running (SSE handles status updates)
  useEffect(() => {
    if (testRun?.status === 'running') {
      // Poll more frequently for live updates - refetch immediately to get latest data
      const interval = setInterval(() => {
        queryClient.refetchQueries({ queryKey: ['conversation', dataRunId] })
        queryClient.refetchQueries({ queryKey: ['test-results', dataRunId] })
      }, 1000) // Poll every 1 second for live updates
      return () => clearInterval(interval)
    }
  }, [testRun?.status, dataRunId, queryClient])
  
  // Also refresh when progress events indicate activity
  useEffect(() => {
    if (progress) {
      console.log('[TestRunDetailPage] Progress event received:', progress.event_type, progress)
      
      // Handle analysis progress events
      if (progress.event_type === 'analysis_started') {
        console.log('[TestRunDetailPage] Analysis started event')
        setAnalysisProgress({ step: 'started', message: progress.message || 'Analysis started...' })
        queryClient.refetchQueries({ queryKey: ['assessments', dataRunId] })
        queryClient.refetchQueries({ queryKey: ['test-run', testRunId] })
      } else if (progress.event_type === 'analysis_progress') {
        const data = progress.data || {}
        console.log('[TestRunDetailPage] Analysis progress event:', data)
        setAnalysisProgress({ 
          step: data.step || 'progress', 
          message: data.message || progress.message || 'Analysis in progress...' 
        })
        queryClient.refetchQueries({ queryKey: ['assessments', dataRunId] })
        queryClient.refetchQueries({ queryKey: ['test-run', testRunId] })
      } else if (progress.event_type === 'analysis_completed') {
        console.log('[TestRunDetailPage] Analysis completed event')
        setAnalysisProgress(null) // Clear progress when done
        queryClient.refetchQueries({ queryKey: ['assessments', dataRunId] })
        queryClient.refetchQueries({ queryKey: ['test-run', testRunId] })
        toast.success(progress.message || 'Analysis completed!')
      }
      
      // Handle test run progress events
      if (testRun?.status === 'running') {
        // Refresh conversation and results when we get any progress update
        // Especially for conversation_turn events, refetch immediately
        if (progress.event_type === 'conversation_turn') {
          queryClient.refetchQueries({ queryKey: ['conversation', dataRunId] })
        } else {
          queryClient.invalidateQueries({ queryKey: ['conversation', dataRunId] })
        }
        queryClient.invalidateQueries({ queryKey: ['test-results', dataRunId] })
      }
    }
  }, [progress, testRun?.status, testRunId, dataRunId, queryClient])

  // Poll for assessments when analysis might be running
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    if (runAnalysis.isPending || analysisProgress || (assessments.length === 0 && testRun?.status === 'completed')) {
      // Poll more aggressively when analysis is pending or in progress
      interval = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ['assessments', dataRunId] })
        queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
      }, 2000) // Check every 2 seconds
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [runAnalysis.isPending, analysisProgress, testRunId, dataRunId, queryClient, assessments.length, testRun?.status])

  const handleRunAnalysis = async (config: AnalysisConfig) => {
    if (!testRunId) {
      toast.error('Invalid test run ID')
      return
    }
    try {
      console.log('Starting analysis with config:', config)
      // Set initial progress state immediately
      setAnalysisProgress({ step: 'starting', message: 'Starting analysis...' })
      const result = await runAnalysis.mutateAsync({
        testRunId,
        config: {
          evaluator_provider: config.evaluator_provider,
          evaluator_model: config.evaluator_model,
          enable_cot_detection: config.enable_cot_detection,
          cot_analysis_mode: config.cot_analysis_mode,
          enable_factuality_check: config.enable_factuality_check,
          enable_manipulation_analysis: config.enable_manipulation_analysis,
        },
      })
      console.log('Analysis started, result:', result)
      
      // Invalidate test runs to show the new analysis test run
      queryClient.invalidateQueries({ queryKey: ['test-runs'] })
      
      // If analysis_test_run_id is returned, navigate to it or show a link
      if (result.analysis_test_run_id) {
        toast.success(`Analysis started! View progress at test run #${result.analysis_test_run_id}`, {
          action: {
            label: 'View',
            onClick: () => router.push(`/test-runs/${result.analysis_test_run_id}`)
          }
        })
        // Optionally navigate to the analysis test run
        // router.push(`/test-runs/${result.analysis_test_run_id}`)
      } else {
        toast.success('Analysis started! Results will appear when complete.')
      }
      
      // Start polling for results
      queryClient.invalidateQueries({ queryKey: ['assessments', dataRunId] })
    } catch (error: any) {
      console.error('Failed to run analysis:', error)
      const errorMessage = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Failed to run analysis'
      toast.error(`Failed to run analysis: ${errorMessage}`)
      // Clear progress on error
      setAnalysisProgress(null)
    }
  }

  const handleStartRun = async () => {
    if (!testRunId) return
    try {
      await startTestRun.mutateAsync(testRunId)
      toast.success('Test run started!')
    } catch (error) {
      console.error('Failed to start test run:', error)
      toast.error('Failed to start test run')
    }
  }

  const handlePauseRun = async () => {
    if (!testRunId) return
    try {
      await pauseTestRun.mutateAsync(testRunId)
      toast.success('Test run paused. You can resume it later.')
      await queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
      await queryClient.refetchQueries({ queryKey: ['test-run', testRunId] })
    } catch (error: any) {
      console.error('Failed to pause test run:', error)
      const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to pause test run'
      toast.error(errorMessage)
    }
  }

  const handleResumeRun = async () => {
    if (!testRunId) return
    try {
      await resumeTestRun.mutateAsync(testRunId)
      toast.success('Test run resumed!')
      await queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
      await queryClient.refetchQueries({ queryKey: ['test-run', testRunId] })
    } catch (error: any) {
      console.error('Failed to resume test run:', error)
      const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to resume test run'
      toast.error(errorMessage)
    }
  }

  const handleStopRun = async () => {
    if (!testRunId) return
    if (!confirm('Are you sure you want to stop this test run? It will be cancelled immediately and can then be deleted.')) {
      return
    }
    try {
      await stopTestRun.mutateAsync(testRunId)
      toast.success('Test run stopped. You can now delete it.')
      // Refresh to get updated status - invalidate and refetch to ensure we get the latest data
      await queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
      await queryClient.refetchQueries({ queryKey: ['test-run', testRunId] })
    } catch (error: any) {
      console.error('Failed to stop test run:', error)
      const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to stop test run'
      toast.error(errorMessage)
    }
  }

  const handleDelete = async () => {
    if (!testRunId) return
    
    // If running, stop it first (this should be rare now since stop button handles it)
    if (testRun?.status === 'running') {
      if (confirm('This test run is currently running. It will be stopped and then deleted. Continue?')) {
        try {
          // Stop the test run first
          await stopTestRun.mutateAsync(testRunId)
          // Wait a moment for status to update
          await new Promise(resolve => setTimeout(resolve, 500))
          // Refresh to get updated status
          queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
          // Then delete
          await deleteTestRun.mutateAsync(testRunId)
          toast.success('Test run stopped and deleted successfully')
          router.push('/test-runs')
        } catch (error: any) {
          const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to stop/delete test run'
          toast.error(errorMessage)
          console.error('Failed to stop/delete test run:', error)
        }
      }
      return
    }
    
    // For cancelled or other non-running tests, delete immediately
    const confirmMessage = testRun?.meta_data?.cancelled 
      ? 'Are you sure you want to delete this cancelled test run? This action cannot be undone.'
      : 'Are you sure you want to delete this test run? This action cannot be undone.'
    
    if (confirm(confirmMessage)) {
      try {
        await deleteTestRun.mutateAsync(testRunId)
        toast.success('Test run deleted successfully')
        router.push('/test-runs')
      } catch (error: any) {
        const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to delete test run'
        toast.error(errorMessage)
        console.error('Failed to delete test run:', error)
      }
    }
  }

  if (loadingRun) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </Layout>
    )
  }

  if (runError || !testRun) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-2">Error loading test run</h2>
            <p className="text-muted-foreground">
              {runError instanceof Error ? runError.message : 'Test run not found'}
            </p>
          </div>
        </div>
      </Layout>
    )
  }

  const resultsColumns = [
      { key: 'test_name', header: 'Test Name' },
      { key: 'test_category', header: 'Category' },
      {
        key: 'score',
        header: 'Score',
        render: (result: TestResult) => (
          <span className="font-medium">
            {result.score !== null && result.score !== undefined
              ? (result.score * 100).toFixed(1) + '%'
              : 'N/A'}
          </span>
        ),
      },
      {
        key: 'flags',
        header: 'Flags',
        render: (result: TestResult) => (
          <div className="flex gap-1">
            {result.flags?.map((flag, i) => (
              <span
                key={i}
                className="rounded bg-red-100 px-2 py-0.5 text-xs text-red-800"
              >
                {flag}
              </span>
            ))}
          </div>
        ),
      },
      {
        key: 'actions',
        header: 'Actions',
        render: (result: TestResult) => (
          <button
            onClick={() => {
              // Expand to show full details
              alert(`Input: ${result.input_prompt}\n\nOutput: ${result.output_response}`)
            }}
            className="text-primary hover:underline text-sm"
          >
            View Details
          </button>
        ),
      },
    ]

  // Check if this is an analysis test run
  const isAnalysisRun = testRun.test_type === 'analysis'
  const sourceTestRunId = testRun.meta_data?.source_test_run_id

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">
              {isAnalysisRun ? 'Analysis' : 'Test Run'} #{testRun.id}
            </h1>
            {isAnalysisRun && sourceTestRunId && (
              <p className="text-sm text-muted-foreground mt-1">
                Analysis of{' '}
                <Link 
                  href={`/test-runs/${sourceTestRunId}`}
                  className="text-primary hover:underline"
                >
                  test run #{sourceTestRunId}
                </Link>
                {' '}— conversation, results, and assessment below are from that run
              </p>
            )}
            <div className="mt-2 flex items-center gap-3">
              <StatusBadge status={testRun.status} />
              <span className="text-sm text-muted-foreground">
                Created: {testRun.created_at ? formatDateTime(testRun.created_at) : 'N/A'}
              </span>
            </div>
          </div>
          <div className="flex gap-2">
            {(testRun.status === 'pending' || (testRun.status === 'failed' && !testRun.meta_data?.cancelled)) && (
              <button
                onClick={handleStartRun}
                disabled={startTestRun.isPending}
                className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                {startTestRun.isPending ? 'Starting...' : 'Start Run'}
              </button>
            )}
            {testRun.status === 'running' && (
              <>
                <button
                  onClick={handlePauseRun}
                  disabled={pauseTestRun.isPending}
                  className="flex items-center gap-2 rounded-lg border border-blue-500 bg-blue-50 px-4 py-2 text-sm font-medium text-blue-700 hover:bg-blue-100 dark:bg-blue-950 dark:text-blue-300 dark:hover:bg-blue-900 disabled:opacity-50"
                >
                  <Pause className="h-4 w-4" />
                  {pauseTestRun.isPending ? 'Pausing...' : 'Pause'}
                </button>
                <button
                  onClick={handleStopRun}
                  disabled={stopTestRun.isPending}
                  className="flex items-center gap-2 rounded-lg border border-orange-500 bg-orange-50 px-4 py-2 text-sm font-medium text-orange-700 hover:bg-orange-100 dark:bg-orange-950 dark:text-orange-300 dark:hover:bg-orange-900 disabled:opacity-50"
                >
                  <Square className="h-4 w-4" />
                  {stopTestRun.isPending ? 'Stopping...' : 'Stop Run'}
                </button>
              </>
            )}
            {testRun.status === 'paused' && (
              <>
                <button
                  onClick={handleResumeRun}
                  disabled={resumeTestRun.isPending}
                  className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                >
                  <RotateCw className="h-4 w-4" />
                  {resumeTestRun.isPending ? 'Resuming...' : 'Resume'}
                </button>
                <button
                  onClick={handleStopRun}
                  disabled={stopTestRun.isPending}
                  className="flex items-center gap-2 rounded-lg border border-orange-500 bg-orange-50 px-4 py-2 text-sm font-medium text-orange-700 hover:bg-orange-100 dark:bg-orange-950 dark:text-orange-300 dark:hover:bg-orange-900 disabled:opacity-50"
                >
                  <Square className="h-4 w-4" />
                  {stopTestRun.isPending ? 'Stopping...' : 'Stop Run'}
                </button>
              </>
            )}
            {testRun.status === 'failed' && testRun.meta_data?.cancelled && (
              <button
                onClick={handleDelete}
                disabled={deleteTestRun.isPending}
                className="flex items-center gap-2 rounded-lg border border-destructive bg-destructive/10 px-4 py-2 text-sm font-medium text-destructive hover:bg-destructive/20 disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" />
                {deleteTestRun.isPending ? 'Deleting...' : 'Delete Run'}
              </button>
            )}
            {testRun.status === 'completed' && !isAnalysisRun && (
              <>
                <button
                  onClick={() => {
                    console.log('Run Analysis button clicked')
                    setShowAnalysisDialog(true)
                  }}
                  disabled={runAnalysis.isPending}
                  className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50"
                >
                  <Play className="h-4 w-4" />
                  {runAnalysis.isPending ? 'Running...' : 'Run Analysis'}
                </button>
                <a
                  href={`/analysis/${testRunId}`}
                  className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
                >
                  View Analysis
                </a>
              </>
            )}
            <button className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted">
              <Download className="h-4 w-4" />
              Export
            </button>
            {testRun.status !== 'running' && !(testRun.status === 'failed' && testRun.meta_data?.cancelled) && (
              <button
                onClick={handleDelete}
                disabled={deleteTestRun.isPending}
                className="flex items-center gap-2 rounded-lg border border-destructive px-4 py-2 text-sm font-medium text-destructive hover:bg-destructive/10 disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" />
                {deleteTestRun.isPending ? 'Deleting...' : 'Delete'}
              </button>
            )}
          </div>
        </div>

        <Tabs.Root value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <Tabs.List className="flex gap-2 border-b">
            <Tabs.Trigger
              value="overview"
              className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
            >
              Overview
            </Tabs.Trigger>
            {(testRun.status === 'running' || testRun.status === 'pending') && (
              <Tabs.Trigger
                value="live"
                className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary flex items-center gap-2"
              >
                <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                Live
              </Tabs.Trigger>
            )}
            <Tabs.Trigger
              value="conversation"
              className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
            >
              Conversation {testRun.status === 'running' && conversation.length > 0 && `(${conversation.length})`}
            </Tabs.Trigger>
            <Tabs.Trigger
              value="results"
              className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
            >
              Results {testRun.status === 'running' && results.length > 0 && `(${results.length})`}
            </Tabs.Trigger>
            {/* Only show Analysis tab after test is completed (or when this run is the analysis run) */}
            {showAnalysisTab && (
              <Tabs.Trigger
                value="analysis"
                className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
              >
                Analysis
              </Tabs.Trigger>
            )}
          </Tabs.List>

          <Tabs.Content value="overview" className="space-y-4">
            {/* Analysis Progress Indicator - Only when this run is the analysis run and it's running, or user just triggered analysis */}
            {((isAnalysisRun && testRun.status === 'running') || (testRun.status === 'completed' && analysisProgress)) && (
              <div className="rounded-lg border bg-purple-50 dark:bg-purple-950 p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-2 rounded-full bg-purple-500 animate-pulse" />
                    <span className="text-sm font-medium text-purple-900 dark:text-purple-100">
                      Analysis in Progress
                    </span>
                  </div>
                  {progress && (
                    <span className="text-xs text-muted-foreground">
                      {new Date(progress.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
                {analysisProgress ? (
                  <>
                    <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
                      {analysisProgress.message || 'Processing...'}
                    </p>
                    {analysisProgress.step && (
                      <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                        Step: {analysisProgress.step.replace(/_/g, ' ')}
                      </p>
                    )}
                  </>
                ) : (
                  <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
                    Analysis is running... waiting for progress updates
                  </p>
                )}
              </div>
            )}
            
            {/* Live Progress Indicator */}
            {(testRun.status === 'running' || testRun.status === 'pending') && !isAnalysisRun && (
              <div className="rounded-lg border bg-blue-50 dark:bg-blue-950 p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                    <span className="text-sm font-medium">
                      {isConnected ? 'Live monitoring active' : progressError ? `Error: ${progressError}` : 'Connecting...'}
                    </span>
                  </div>
                  {progress && (
                    <span className="text-xs text-muted-foreground">
                      {new Date(progress.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
                {progressError && (
                  <div className="mt-2 text-xs text-red-600 dark:text-red-400">
                    {progressError}
                  </div>
                )}
                {progress && (
                  <div className="mt-2 space-y-2">
                    {progress.message && (
                      <p className="text-sm text-blue-900 dark:text-blue-100">{progress.message}</p>
                    )}
                    
                    {/* Progress bar for benchmarks */}
                    {progress.data && progress.event_type === 'benchmark_progress' && typeof progress.data.progress === 'number' && (
                      <div className="mt-2">
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span>Progress</span>
                          <span>{progress.data.progress}%</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-blue-200 dark:bg-blue-800 overflow-hidden">
                          <div
                            className="h-full bg-blue-600 dark:bg-blue-400 transition-all duration-300"
                            style={{ width: `${Math.min(100, Math.max(0, progress.data.progress))}%` }}
                          />
                        </div>
                        {progress.data.current && progress.data.total && (
                          <p className="text-xs text-muted-foreground mt-1">
                            Question {progress.data.current} of {progress.data.total}
                          </p>
                        )}
                      </div>
                    )}
                    
                    {/* Progress bar for conversation tests - Enhanced */}
                    {progress.data && progress.event_type === 'conversation_progress' && typeof progress.data.progress === 'number' && (
                      <div className="mt-3 p-3 rounded-lg border bg-white dark:bg-gray-800">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-semibold">Conversation Progress</span>
                          <span className="text-sm font-bold text-green-600 dark:text-green-400">
                            {progress.data.progress}%
                          </span>
                        </div>
                        <div className="h-3 w-full rounded-full bg-green-200 dark:bg-green-800 overflow-hidden mb-2">
                          <div
                            className="h-full bg-green-600 dark:bg-green-400 transition-all duration-300"
                            style={{ width: `${Math.min(100, Math.max(0, progress.data.progress))}%` }}
                          />
                        </div>
                        {progress.data.turn_number !== undefined && progress.data.max_turns && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">
                              Turn {progress.data.turn_number} of {progress.data.max_turns}
                              {progress.data.speaker && ` • ${progress.data.speaker}`}
                            </span>
                            {progress.data.total_turns && (
                              <span className="text-muted-foreground">
                                {progress.data.total_turns} total turns
                              </span>
                            )}
                          </div>
                        )}
                        {progress.data.response_preview && (
                          <div className="mt-2 p-2 rounded bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800">
                            <p className="text-xs font-semibold text-green-700 dark:text-green-300 mb-1">Latest Response:</p>
                            <p className="text-xs text-gray-700 dark:text-gray-300">
                              {progress.data.response_preview}
                            </p>
                          </div>
                        )}
                        {progress.message && (
                          <p className="text-xs text-muted-foreground mt-2 italic">
                            {progress.message}
                          </p>
                        )}
                      </div>
                    )}
                    
                    {/* Test progress with attempt count - Enhanced for visibility */}
                    {progress.data && progress.event_type === 'test_progress' && (
                      <div className="mt-3 p-3 rounded-lg border bg-white dark:bg-gray-800">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-semibold">
                            {progress.data.test_type === 'one_shot' ? 'One-Shot Test' : 
                             progress.data.test_type === 'multi_shot' ? 'Multi-Shot Test' : 
                             'Test Progress'}
                          </span>
                          {typeof progress.data.progress === 'number' && (
                            <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
                              {progress.data.progress}%
                            </span>
                          )}
                        </div>
                        {typeof progress.data.progress === 'number' && (
                          <div className="h-3 w-full rounded-full bg-blue-200 dark:bg-blue-800 overflow-hidden mb-2">
                            <div
                              className="h-full bg-blue-600 dark:bg-blue-400 transition-all duration-300"
                              style={{ width: `${Math.min(100, Math.max(0, progress.data.progress))}%` }}
                            />
                          </div>
                        )}
                        {typeof progress.data.attempt === 'number' && typeof progress.data.total_attempts === 'number' && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">
                              Processing attempt {progress.data.attempt} of {progress.data.total_attempts}
                            </span>
                            {progress.data.current && progress.data.total && (
                              <span className="text-muted-foreground">
                                {progress.data.current}/{progress.data.total}
                              </span>
                            )}
                          </div>
                        )}
                        {progress.message && (
                          <p className="text-xs text-muted-foreground mt-1 italic">
                            {progress.message}
                          </p>
                        )}
                      </div>
                    )}

                    {/* General progress indicator for any running test */}
                    {testRun?.status === 'running' && progress.data && typeof progress.data.elapsed_seconds === 'number' && 
                     progress.event_type !== 'test_progress' && (
                      <div className="mt-2">
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span>Status</span>
                          <span className="text-green-600 font-medium">Running</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Elapsed: {Math.floor(progress.data.elapsed_seconds)}s
                        </p>
                      </div>
                    )}
                    
                    {/* Fallback: Show elapsed time if available */}
                    {progress.data && typeof progress.data.elapsed_seconds === 'number' && 
                     progress.event_type !== 'test_progress' && 
                     progress.event_type !== 'conversation_progress' && 
                     progress.event_type !== 'benchmark_progress' && (
                      <p className="text-xs text-muted-foreground">
                        Elapsed: {Math.floor(progress.data.elapsed_seconds)}s
                      </p>
                    )}
                  </div>
                )}
                
                {/* Show activity indicator even when no progress data */}
                {testRun?.status === 'running' && !progress && (
                  <div className="mt-2">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                      <p className="text-xs text-muted-foreground">Test is running... waiting for updates</p>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border bg-card p-4">
                <h3 className="font-semibold mb-2">{isAnalysisRun ? 'Evaluator Model' : 'Doctor Model'}</h3>
                <p className="text-sm text-muted-foreground">{testRun.doctor_provider}</p>
                <p className="text-lg font-medium">{testRun.doctor_model}</p>
              </div>
              {testRun.test_type === 'group_therapy' && testRun.meta_data?.patients ? (
                <div className="rounded-lg border bg-card p-4 col-span-2">
                  <h3 className="font-semibold mb-2">Patient Models ({testRun.meta_data.patients.length})</h3>
                  <div className="grid grid-cols-2 gap-3 mt-3">
                    {testRun.meta_data.patients.map((patient: any, index: number) => (
                      <div key={index} className="rounded border bg-muted/30 p-3">
                        <p className="text-xs text-muted-foreground">Patient {index + 1}</p>
                        <p className="text-sm font-medium">{patient.provider}</p>
                        <p className="text-base font-semibold">{patient.model}</p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
              <div className="rounded-lg border bg-card p-4">
                <h3 className="font-semibold mb-2">{isAnalysisRun ? 'Model Being Analyzed' : 'Patient Model'}</h3>
                <p className="text-sm text-muted-foreground">{testRun.patient_provider}</p>
                <p className="text-lg font-medium">{testRun.patient_model}</p>
              </div>
              )}
              <div className="rounded-lg border bg-card p-4">
                <h3 className="font-semibold mb-2">Test Type</h3>
                <p className="text-lg font-medium capitalize">{testRun.test_type}</p>
              </div>
              <div className="rounded-lg border bg-card p-4">
                <h3 className="font-semibold mb-2">Status</h3>
                <StatusBadge status={testRun.status} />
              </div>
            </div>
          </Tabs.Content>

          {/* Live Monitoring Tab */}
          {(testRun.status === 'running' || testRun.status === 'pending') && (
            <Tabs.Content value="live" className="space-y-4">
              {/* Status Bar */}
              <div className="rounded-lg border bg-card p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                    <span className="text-sm font-medium">
                      {isConnected ? 'Live' : progressError || 'Connecting...'}
                    </span>
                    {progress && typeof progress.data?.elapsed_seconds === 'number' && (
                      <span className="text-xs text-muted-foreground">
                        {Math.floor(progress.data.elapsed_seconds)}s elapsed
                      </span>
                    )}
                  </div>
                  {conversation.length > 0 && (
                    <span className="text-xs text-muted-foreground">
                      {conversation.length} {conversation.length === 1 ? 'message' : 'messages'}
                    </span>
                  )}
                </div>
              </div>

              {/* Live Messages Display - Chat-like format */}
              <div className="rounded-lg border bg-card p-4">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  Live Messages
                  {testRun.status === 'running' && (
                    <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                  )}
                </h3>
                
                {conversation.length > 0 ? (
                  <div className="space-y-4 max-h-[600px] overflow-y-auto" id="live-messages-container">
                    {conversation.map((turn) => {
                      const isDoctor = turn.speaker === 'doctor'
                      const patientName = turn.metadata?.patient_name
                      const patientModel = turn.metadata?.patient_model
                      const reasoning = turn.metadata?.reasoning
                      const hasPrompt = turn.prompt && turn.prompt.trim().length > 0
                      // Context is what this speaker is replying to (the other party). Label the other speaker.
                      const replyingToLabel = isDoctor ? '🤖 Patient' : '👨‍⚕️ Doctor'

                      return (
                        <div key={turn.id} className="space-y-1">
                          {/* Compact context: what this AI is responding to (no duplicate full message) */}
                          {hasPrompt && (
                            <div className="text-xs text-muted-foreground italic border-l-2 border-muted pl-2 ml-2">
                              <span className="font-medium">Replying to {replyingToLabel}: </span>
                              <span>{turn.prompt.length > 180 ? `${turn.prompt.substring(0, 180)}…` : turn.prompt}</span>
                            </div>
                          )}
                          {/* Main message: one bubble per turn */}
                          {turn.response && (
                            <div className={`flex ${isDoctor ? 'justify-start' : 'justify-end'}`}>
                              <div className={`max-w-[80%] rounded-2xl px-4 py-3 shadow-sm ${
                                isDoctor
                                  ? 'bg-blue-500 text-white'
                                  : 'bg-green-500 text-white'
                              }`}>
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="text-xs font-semibold opacity-90">
                                    {isDoctor ? '👨‍⚕️ Doctor' : (patientName ? `🤖 ${patientName}${patientModel ? ` (${patientModel})` : ''}` : '🤖 Patient')}
                                  </span>
                                  <span className="text-xs opacity-75">
                                    Turn {turn.turn_number}
                                  </span>
                                </div>
                                {reasoning && (
                                  <div className="mb-2 p-2 rounded bg-black/10 dark:bg-white/10">
                                    <p className="text-xs font-semibold mb-1 opacity-90">💭 Reasoning:</p>
                                    <p className="text-xs whitespace-pre-wrap opacity-90">{reasoning}</p>
                                  </div>
                                )}
                                <p className="text-sm whitespace-pre-wrap">{turn.response}</p>
                                {turn.created_at && (
                                  <p className="text-xs opacity-75 mt-1">
                                    {new Date(turn.created_at).toLocaleTimeString()}
                                  </p>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <div className="text-center text-muted-foreground py-12">
                    <div className="flex flex-col items-center gap-2">
                      <div className="h-8 w-8 rounded-full border-2 border-muted-foreground/30 border-t-primary animate-spin" />
                      <p className="text-sm font-medium">Waiting for messages...</p>
                      {testRun.status === 'running' && (
                        <p className="text-xs mt-1">
                          Messages will appear here as they're sent
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Progress Summary (collapsed) */}
              {progress && progress.data && (
                <details className="rounded-lg border bg-card p-3">
                  <summary className="text-sm font-medium cursor-pointer">Progress Details</summary>
                  <div className="mt-3 space-y-2">
                    {progress.event_type === 'benchmark_progress' && typeof progress.data.progress === 'number' && (
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span>Benchmark Progress</span>
                          <span>{progress.data.progress}%</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full bg-primary transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(0, progress.data.progress))}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {progress.event_type === 'test_progress' && typeof progress.data.progress === 'number' && (
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span>Test Progress</span>
                          <span>{progress.data.progress}%</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full bg-primary transition-all duration-500"
                            style={{ width: `${Math.min(100, Math.max(0, progress.data.progress))}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {progress.message && (
                      <p className="text-xs text-muted-foreground">{progress.message}</p>
                    )}
                  </div>
                </details>
              )}

            </Tabs.Content>
          )}

          <Tabs.Content value="conversation">
            {loadingConv ? (
              <LoadingSpinner />
            ) : convError ? (
              <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
                Error loading conversation: {convError instanceof Error ? convError.message : 'Unknown error'}
              </div>
            ) : (
              <div className="space-y-4">
                {testRun.status === 'running' && (
                  <div className="rounded-lg border bg-blue-50 dark:bg-blue-950 p-3 flex items-center gap-2">
                    <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-sm text-blue-900 dark:text-blue-100">
                      Live updates enabled - conversation turns will appear here as they're generated
                    </span>
                  </div>
                )}
                <ConversationViewer turns={conversation} />
              </div>
            )}
          </Tabs.Content>

          <Tabs.Content value="results">
            {loadingResults ? (
              <LoadingSpinner />
            ) : resultsError ? (
              <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
                Error loading results: {resultsError instanceof Error ? resultsError.message : 'Unknown error'}
              </div>
            ) : (
              <div className="space-y-4">
                <DataTable
                  data={results}
                  columns={resultsColumns}
                  emptyMessage="No results available"
                />
              </div>
            )}
          </Tabs.Content>

          <Tabs.Content value="analysis" className="overflow-y-auto max-h-[calc(100vh-300px)]">
            {/* Analysis Progress Indicator */}
            {analysisProgress && (
              <div className="mb-4 rounded-lg border bg-purple-50 dark:bg-purple-950 p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-2 rounded-full bg-purple-500 animate-pulse" />
                    <span className="text-sm font-medium text-purple-900 dark:text-purple-100">
                      Analysis in Progress
                    </span>
                  </div>
                  {progress && (
                    <span className="text-xs text-muted-foreground">
                      {new Date(progress.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
                <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
                  {analysisProgress.message || 'Processing...'}
                </p>
                {analysisProgress.step && (
                  <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                    Step: {analysisProgress.step.replace(/_/g, ' ')}
                  </p>
                )}
              </div>
            )}
            {runAnalysis.isPending && !analysisProgress && (
              <div className="mb-4 rounded-lg border bg-blue-50 p-4 text-sm text-blue-800">
                <p className="font-medium">Analysis is running...</p>
                <p className="text-xs mt-1">This may take a few minutes. Results will appear automatically when complete.</p>
              </div>
            )}
            {loadingAssessments ? (
              <LoadingSpinner />
            ) : assessments.length > 0 ? (
              <div className="space-y-4">
                {assessments.map((assessment) => (
                  <div key={assessment.id} className="rounded-lg border bg-card p-6">
                    <div className="mb-4">
                      <h3 className="font-semibold mb-2">Safety Score</h3>
                      <p className="text-3xl font-bold">
                        {(assessment.safety_score || assessment.overall_score)
                          ? ((assessment.safety_score || assessment.overall_score || 0) * 100).toFixed(1) + '%'
                          : 'N/A'}
                      </p>
                    </div>
                    {assessment.assessment_text && (
                      <div className="mt-4">
                        <h3 className="font-semibold mb-2">Assessment</h3>
                        <p className="text-sm whitespace-pre-wrap">{assessment.assessment_text}</p>
                      </div>
                    )}
                    {assessment.flags && assessment.flags.length > 0 && (
                      <div className="mt-4">
                        <h3 className="font-semibold mb-2">Flags</h3>
                        <div className="flex flex-wrap gap-2">
                          {assessment.flags.map((flag, i) => (
                            <span
                              key={i}
                              className="rounded bg-red-100 px-2 py-1 text-xs text-red-800"
                            >
                              {flag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {assessment.metadata?.cot_analysis && (
                      <div className="mt-4">
                        <h3 className="font-semibold mb-2">Chain of Thought Analysis</h3>
                        <div className="rounded bg-muted/50 p-4">
                          <div className="mb-2">
                            <span className="text-sm font-medium">
                              COT Detected: {assessment.metadata.cot_analysis.cot_detected ? 'Yes' : 'No'}
                            </span>
                            <span className="ml-2 text-xs text-muted-foreground">
                              (Mode: {assessment.metadata.cot_analysis.mode})
                            </span>
                          </div>
                          {assessment.metadata.cot_analysis.ai_analysis && (
                            <div className="mt-2">
                              <p className="text-sm font-medium mb-1">AI Analysis:</p>
                              <p className="text-sm whitespace-pre-wrap text-muted-foreground">
                                {assessment.metadata.cot_analysis.ai_analysis}
                              </p>
                            </div>
                          )}
                          {assessment.metadata.cot_analysis.analysis && !assessment.metadata.cot_analysis.ai_analysis && (
                            <p className="text-sm text-muted-foreground mt-2">
                              {assessment.metadata.cot_analysis.analysis}
                            </p>
                          )}
                        </div>
                      </div>
                    )}
                    {assessment.metadata?.factuality_analysis && (
                      <div className="mt-4">
                        <h3 className="font-semibold mb-2">Factuality Analysis</h3>
                        <div className="rounded bg-muted/50 p-4">
                          <div className="mb-2">
                            <span className="text-sm font-medium">
                              Factuality Score: {(assessment.metadata.factuality_analysis.factuality_score * 100).toFixed(1)}%
                            </span>
                          </div>
                          {assessment.metadata.factuality_analysis.verified_claims && assessment.metadata.factuality_analysis.verified_claims.length > 0 && (
                            <div className="mt-2">
                              <p className="text-sm font-medium mb-1">Verified Claims ({assessment.metadata.factuality_analysis.verified_claims.length}):</p>
                              <ul className="text-sm text-muted-foreground list-disc list-inside">
                                {assessment.metadata.factuality_analysis.verified_claims.slice(0, 5).map((claim, i) => (
                                  <li key={i}>{claim}</li>
                                ))}
                                {assessment.metadata.factuality_analysis.verified_claims.length > 5 && (
                                  <li className="text-xs">... and {assessment.metadata.factuality_analysis.verified_claims.length - 5} more</li>
                                )}
                              </ul>
                            </div>
                          )}
                          {assessment.metadata.factuality_analysis.potential_hallucinations && assessment.metadata.factuality_analysis.potential_hallucinations.length > 0 && (
                            <div className="mt-2">
                              <p className="text-sm font-medium mb-1 text-red-600">Potential Hallucinations ({assessment.metadata.factuality_analysis.potential_hallucinations.length}):</p>
                              <ul className="text-sm text-muted-foreground list-disc list-inside">
                                {assessment.metadata.factuality_analysis.potential_hallucinations.slice(0, 5).map((hall, i) => (
                                  <li key={i}>
                                    {hall.claim || 'Unknown claim'}: {hall.reason || 'Identified as potential hallucination'}
                                  </li>
                                ))}
                                {assessment.metadata.factuality_analysis.potential_hallucinations.length > 5 && (
                                  <li className="text-xs">... and {assessment.metadata.factuality_analysis.potential_hallucinations.length - 5} more</li>
                                )}
                              </ul>
                            </div>
                          )}
                          {assessment.metadata.factuality_analysis.analysis && (
                            <div className="mt-2">
                              <p className="text-sm font-medium mb-1">Analysis:</p>
                              <p className="text-sm whitespace-pre-wrap text-muted-foreground">
                                {assessment.metadata.factuality_analysis.analysis}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    {assessment.metadata?.manipulation_resistance && (
                      <div className="mt-4">
                        <h3 className="font-semibold mb-2">Manipulation Resistance Analysis</h3>
                        <div className="rounded bg-muted/50 p-4">
                          <div className="mb-2">
                            <span className="text-sm font-medium">
                              Resistance Score: {(assessment.metadata.manipulation_resistance.resistance_score * 100).toFixed(1)}%
                            </span>
                            <span className="ml-2 text-xs text-muted-foreground">
                              (Higher is better)
                            </span>
                          </div>
                          <div className="mt-2 text-sm text-muted-foreground">
                            <p>Manipulation Attempts: {assessment.metadata.manipulation_resistance.manipulation_attempts?.length || 0}</p>
                            <p>Successful Resistances: {assessment.metadata.manipulation_resistance.successful_resistances?.length || 0}</p>
                            <p>Failed Resistances: {assessment.metadata.manipulation_resistance.failed_resistances?.length || 0}</p>
                          </div>
                          {assessment.metadata.manipulation_resistance.analysis && (
                            <div className="mt-2">
                              <p className="text-sm font-medium mb-1">Analysis:</p>
                              <p className="text-sm whitespace-pre-wrap text-muted-foreground">
                                {assessment.metadata.manipulation_resistance.analysis}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    {assessment.metadata?.manipulation_capability && (
                      <div className="mt-4">
                        <h3 className="font-semibold mb-2">Manipulation Capability Analysis</h3>
                        <div className="rounded bg-muted/50 p-4">
                          <div className="mb-2">
                            <span className="text-sm font-medium">
                              Capability Score: {(assessment.metadata.manipulation_capability.capability_score * 100).toFixed(1)}%
                            </span>
                            <span className="ml-2 text-xs text-muted-foreground">
                              (Lower is better)
                            </span>
                          </div>
                          <div className="mt-2 text-sm text-muted-foreground">
                            <p>Manipulative Techniques: {assessment.metadata.manipulation_capability.manipulative_techniques?.length || 0}</p>
                            <p>Ethical Violations: {assessment.metadata.manipulation_capability.ethical_violations?.length || 0}</p>
                          </div>
                          {assessment.metadata.manipulation_capability.analysis && (
                            <div className="mt-2">
                              <p className="text-sm font-medium mb-1">Analysis:</p>
                              <p className="text-sm whitespace-pre-wrap text-muted-foreground">
                                {assessment.metadata.manipulation_capability.analysis}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
                <p className="mb-4">No analysis available yet.</p>
                {!isAnalysisRun && (
                  <button
                    onClick={() => setShowAnalysisDialog(true)}
                    disabled={runAnalysis.isPending}
                    className="rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                  >
                    {runAnalysis.isPending ? 'Running Analysis...' : 'Run Analysis'}
                  </button>
                )}
                {runAnalysis.isPending && (
                  <p className="mt-4 text-sm text-muted-foreground">
                    Analysis is running in the background. Results will appear here when complete.
                  </p>
                )}
              </div>
            )}
          </Tabs.Content>
        </Tabs.Root>
      </div>

      {showAnalysisDialog && (
        <AnalysisConfigDialog
          isOpen={true}
          onClose={() => setShowAnalysisDialog(false)}
          onConfirm={handleRunAnalysis}
          doctorProvider={testRun?.doctor_provider}
          doctorModel={testRun?.doctor_model}
          isLoading={runAnalysis.isPending}
        />
      )}
    </Layout>
  )
}
