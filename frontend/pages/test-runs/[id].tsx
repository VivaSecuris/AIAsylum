import { useRouter } from 'next/router'
import { useState, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { ConversationViewer } from '@/components/conversation/ConversationViewer'
import { DataTable } from '@/components/common/DataTable'
import { AnalysisConfigDialog, AnalysisConfig } from '@/components/analysis/AnalysisConfigDialog'
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
  const { data: conversation = [], isLoading: loadingConv, error: convError } = useConversation(testRunId)
  const { data: results = [], isLoading: loadingResults, error: resultsError } = useTestResults(testRunId)
  const { data: assessments = [], isLoading: loadingAssessments } = useAssessments(testRunId)
  const runAnalysis = useRunAnalysis()
  const startTestRun = useStartTestRun()
  const deleteTestRun = useDeleteTestRun()
  
  // Enable refetching when test run is running
  useEffect(() => {
    if (testRun?.status === 'running') {
      const interval = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [testRun?.status, testRunId, queryClient])

  // Poll for assessments when analysis might be running
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    if (runAnalysis.isPending || (assessments.length === 0 && testRun?.status === 'completed')) {
      // Poll more aggressively when analysis is pending
      interval = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ['assessments', testRunId] })
        queryClient.invalidateQueries({ queryKey: ['test-run', testRunId] })
      }, 2000) // Check every 2 seconds
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [runAnalysis.isPending, testRunId, queryClient, assessments.length, testRun?.status])

  const [activeTab, setActiveTab] = useState('overview')
  const [showAnalysisDialog, setShowAnalysisDialog] = useState(false)

  const handleRunAnalysis = async (config: AnalysisConfig) => {
    if (!testRunId) {
      toast.error('Invalid test run ID')
      return
    }
    try {
      console.log('Starting analysis with config:', config)
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
      toast.success('Analysis started! Results will appear when complete.')
      // Start polling for results
      queryClient.invalidateQueries({ queryKey: ['assessments', testRunId] })
    } catch (error: any) {
      console.error('Failed to run analysis:', error)
      const errorMessage = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Failed to run analysis'
      toast.error(`Failed to run analysis: ${errorMessage}`)
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
                  <h3 className="font-semibold mb-2">Patient Model</h3>
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

          <Tabs.Content value="analysis" className="overflow-y-auto max-h-[calc(100vh-300px)]">
            {runAnalysis.isPending && (
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
                <button
                  onClick={() => setShowAnalysisDialog(true)}
                  disabled={runAnalysis.isPending}
                  className="rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                >
                  {runAnalysis.isPending ? 'Running Analysis...' : 'Run Analysis'}
                </button>
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

      <AnalysisConfigDialog
        isOpen={showAnalysisDialog}
        onClose={() => {
          console.log('Dialog closed')
          setShowAnalysisDialog(false)
        }}
        onConfirm={handleRunAnalysis}
        doctorProvider={testRun?.doctor_provider}
        doctorModel={testRun?.doctor_model}
        isLoading={runAnalysis.isPending}
      />
    </Layout>
  )
}
