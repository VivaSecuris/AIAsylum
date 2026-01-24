import { useRouter } from 'next/router'
import { useState, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { ConversationViewer } from '@/components/conversation/ConversationViewer'
import { DataTable } from '@/components/common/DataTable'
import { useTestRun, useConversation, useTestResults, useAssessments, useRunAnalysis, useStartTestRun, useDeleteTestRun } from '@/lib/hooks'
import { formatDate } from '@/lib/utils'
import { TestResult, Assessment } from '@/lib/api'
import { Play, Download, GitCompare, Trash2 } from 'lucide-react'
import * as Tabs from '@radix-ui/react-tabs'
import { toast } from '@/lib/toast'

export default function TestRunDetailPage() {
  const router = useRouter()
  const { id } = router.query
  const testRunId = typeof id === 'string' ? parseInt(id) : 0

  const queryClient = useQueryClient()
  const { data: testRun, isLoading: loadingRun, error: runError } = useTestRun(testRunId)
  
  // Enable refetching when test run is running
  useEffect(() => {
    if (testRun?.status === 'running') {
      const interval = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [testRun?.status, testRunId, queryClient])
  const { data: conversation = [], isLoading: loadingConv, error: convError } = useConversation(testRunId)
  const { data: results = [], isLoading: loadingResults, error: resultsError } = useTestResults(testRunId)
  const { data: assessments = [], isLoading: loadingAssessments } = useAssessments(testRunId)
  const runAnalysis = useRunAnalysis()
  const startTestRun = useStartTestRun()
  const deleteTestRun = useDeleteTestRun()

  const [activeTab, setActiveTab] = useState('overview')

  const handleRunAnalysis = async () => {
    if (!testRunId) return
    try {
      await runAnalysis.mutateAsync({
        testRunId,
        config: {
          enable_cot_detection: true,
          cot_analysis_mode: 'full',
        },
      })
      toast.success('Analysis started!')
    } catch (error) {
      console.error('Failed to run analysis:', error)
      toast.error('Failed to run analysis')
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

  const handleDelete = async () => {
    if (!testRunId) return
    if (testRun?.status === 'running') {
      toast.error('Cannot delete a test run that is currently running')
      return
    }
    if (confirm('Are you sure you want to delete this test run? This action cannot be undone.')) {
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

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Test Run #{testRun.id}</h1>
            <div className="mt-2 flex items-center gap-3">
              <StatusBadge status={testRun.status} />
              <span className="text-sm text-muted-foreground">
                Created: {testRun.created_at ? formatDate(testRun.created_at) : 'N/A'}
              </span>
            </div>
          </div>
          <div className="flex gap-2">
            {(testRun.status === 'pending' || testRun.status === 'failed') && (
              <button
                onClick={handleStartRun}
                disabled={startTestRun.isPending}
                className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                {startTestRun.isPending ? 'Starting...' : 'Start Run'}
              </button>
            )}
            {testRun.status === 'completed' && (
              <>
                <button
                  onClick={handleRunAnalysis}
                  disabled={runAnalysis.isPending}
                  className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
                >
                  <Play className="h-4 w-4" />
                  Run Analysis
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
            {testRun.status !== 'running' && (
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
            <Tabs.Trigger
              value="conversation"
              className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
            >
              Conversation
            </Tabs.Trigger>
            <Tabs.Trigger
              value="results"
              className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
            >
              Results
            </Tabs.Trigger>
            <Tabs.Trigger
              value="analysis"
              className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
            >
              Analysis
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="overview" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border bg-card p-4">
                <h3 className="font-semibold mb-2">Doctor Model</h3>
                <p className="text-sm text-muted-foreground">{testRun.doctor_provider}</p>
                <p className="text-lg font-medium">{testRun.doctor_model}</p>
              </div>
              <div className="rounded-lg border bg-card p-4">
                <h3 className="font-semibold mb-2">Patient Model</h3>
                <p className="text-sm text-muted-foreground">{testRun.patient_provider}</p>
                <p className="text-lg font-medium">{testRun.patient_model}</p>
              </div>
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

          <Tabs.Content value="conversation">
            {loadingConv ? (
              <LoadingSpinner />
            ) : convError ? (
              <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
                Error loading conversation: {convError instanceof Error ? convError.message : 'Unknown error'}
              </div>
            ) : (
              <ConversationViewer turns={conversation} />
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
              <DataTable
                data={results}
                columns={resultsColumns}
                emptyMessage="No results available"
              />
            )}
          </Tabs.Content>

          <Tabs.Content value="analysis">
            {loadingAssessments ? (
              <LoadingSpinner />
            ) : assessments.length > 0 ? (
              <div className="space-y-4">
                {assessments.map((assessment) => (
                  <div key={assessment.id} className="rounded-lg border bg-card p-6">
                    <div className="mb-4">
                      <h3 className="font-semibold mb-2">Safety Score</h3>
                      <p className="text-3xl font-bold">
                        {assessment.safety_score
                          ? (assessment.safety_score * 100).toFixed(1) + '%'
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
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
                <p>No analysis available yet.</p>
                <button
                  onClick={handleRunAnalysis}
                  className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
                >
                  Run Analysis
                </button>
              </div>
            )}
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </Layout>
  )
}
