import { useRouter } from 'next/router'
import { useState, useMemo, useEffect } from 'react'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { SafetyScoreCard } from '@/components/analysis/SafetyScoreCard'
import { AnalysisConfigDialog, AnalysisConfig } from '@/components/analysis/AnalysisConfigDialog'
import { useQueryClient } from '@tanstack/react-query'
import { useAssessments, useRunAnalysis, useTestRun, useTestResults } from '@/lib/hooks'
import { formatDate } from '@/lib/utils'
import { Play } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, ComposedChart, Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts'
import { TEST_TYPES } from '@/lib/create-test-config'
import { toast } from '@/lib/toast'

export default function AnalysisPage() {
  const router = useRouter()
  const { testRunId } = router.query
  const id = typeof testRunId === 'string' ? parseInt(testRunId) : 0

  const queryClient = useQueryClient()
  const { data: testRun, isLoading: loadingRun } = useTestRun(id)
  // For analysis runs, assessments and results belong to the source run
  const dataRunId: number = (testRun?.test_type === 'analysis' && testRun?.meta_data?.source_test_run_id)
    ? testRun.meta_data.source_test_run_id
    : id
  const { data: assessments = [], isLoading } = useAssessments(dataRunId)
  const { data: testResults = [] } = useTestResults(dataRunId)
  const runAnalysis = useRunAnalysis()
  const [showAnalysisDialog, setShowAnalysisDialog] = useState(false)

  // Poll for assessments when analysis might be running
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    if (runAnalysis.isPending || (assessments.length === 0 && testRun?.status === 'completed')) {
      // Poll more aggressively when analysis is pending
      interval = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ['assessments', dataRunId] })
        queryClient.invalidateQueries({ queryKey: ['test-run', id] })
      }, 2000) // Check every 2 seconds
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [runAnalysis.isPending, id, dataRunId, queryClient, assessments.length, testRun?.status])

  // All useMemo hooks must be declared before any early returns (Rules of Hooks)

  // Score trends across multiple assessments
  const scoreTrendData = useMemo(() => {
    return assessments.map((assessment, index) => ({
      assessment: `Assessment ${index + 1}`,
      date: assessment.created_at ? new Date(assessment.created_at).toLocaleDateString() : `#${index + 1}`,
      safetyScore: (assessment.scores?.safety ?? assessment.overall_score ?? 0) * 100,
      timestamp: assessment.created_at ? new Date(assessment.created_at).getTime() : index,
    })).sort((a, b) => a.timestamp - b.timestamp)
  }, [assessments])

  // Multi-assessment score comparison
  const multiAssessmentData = useMemo(() => {
    if (assessments.length === 0) return []
    const scoreKeys = new Set<string>()
    assessments.forEach(assessment => {
      if (assessment.scores) {
        Object.keys(assessment.scores).forEach(key => scoreKeys.add(key))
      }
    })
    
    return Array.from(scoreKeys).map(key => {
      const values = assessments
        .map(a => a.scores?.[key])
        .filter(v => v !== undefined)
        .map(v => (v as number) * 100)
      return {
        name: key.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
        ...assessments.reduce((acc, assessment, idx) => {
          const value = assessment.scores?.[key]
          acc[`Assessment ${idx + 1}`] = value ? value * 100 : 0
          return acc
        }, {} as Record<string, number>),
        average: values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0,
      }
    })
  }, [assessments])

  // Test results score distribution
  const testResultsData = useMemo(() => {
    const scoreRanges = [
      { name: '0-20%', min: 0, max: 20, count: 0 },
      { name: '21-40%', min: 21, max: 40, count: 0 },
      { name: '41-60%', min: 41, max: 60, count: 0 },
      { name: '61-80%', min: 61, max: 80, count: 0 },
      { name: '81-100%', min: 81, max: 100, count: 0 },
    ]
    
    testResults.forEach(result => {
      const score = result.score !== null && result.score !== undefined ? result.score * 100 : null
      if (score !== null) {
        const range = scoreRanges.find(r => score >= r.min && score <= r.max)
        if (range) range.count++
      }
    })
    
    return scoreRanges
  }, [testResults])

  // Category breakdown from test results
  const categoryData = useMemo(() => {
    const categories: Record<string, { count: number; totalScore: number; scores: number[] }> = {}
    testResults.forEach(result => {
      const category = result.test_category || 'Uncategorized'
      if (!categories[category]) {
        categories[category] = { count: 0, totalScore: 0, scores: [] }
      }
      categories[category].count++
      if (result.score !== null && result.score !== undefined) {
        categories[category].totalScore += result.score * 100
        categories[category].scores.push(result.score * 100)
      }
    })
    
    return Object.entries(categories).map(([name, data]) => ({
      name,
      count: data.count,
      avgScore: data.scores.length > 0 ? data.totalScore / data.scores.length : 0,
    }))
  }, [testResults])

  // Flags distribution
  const flagsData = useMemo(() => {
    const flagCounts: Record<string, number> = {}
    assessments.forEach(assessment => {
      if (assessment.flags) {
        assessment.flags.forEach(flag => {
          flagCounts[flag] = (flagCounts[flag] || 0) + 1
        })
      }
    })
    testResults.forEach(result => {
      if (result.flags) {
        result.flags.forEach(flag => {
          flagCounts[flag] = (flagCounts[flag] || 0) + 1
        })
      }
    })
    
    return Object.entries(flagCounts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10)
  }, [assessments, testResults])

  if (!id || id === 0) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-2">Invalid Test Run ID</h2>
            <p className="text-muted-foreground">Please select a test run to view analysis.</p>
          </div>
        </div>
      </Layout>
    )
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

  const handleRunAnalysis = async (config: AnalysisConfig) => {
    if (!id) {
      toast.error('Invalid test run ID')
      return
    }
    try {
      console.log('Starting analysis with config:', config)
      const result = await runAnalysis.mutateAsync({
        testRunId: id,
        config: { ...config },
      })
      console.log('Analysis started, result:', result)
      toast.success('Analysis started! Results will appear when complete.')
      // Start polling for results
      queryClient.invalidateQueries({ queryKey: ['assessments', id] })
    } catch (error: any) {
      console.error('Failed to run analysis:', error)
      const errorMessage = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Failed to run analysis'
      toast.error(`Failed to run analysis: ${errorMessage}`)
    }
  }

  // Chart data for score breakdown
  const chartData = assessments[0]?.scores
    ? Object.entries(assessments[0].scores).map(([key, value]) => ({
        name: key.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
        value: value * 100,
      }))
    : []

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Analysis Dashboard</h1>
            {testRun && (
              <p className="text-sm text-muted-foreground mt-1">
                Test Run #{testRun.id} • {TEST_TYPES.find((t) => t.value === testRun.test_type)?.label ?? testRun.test_type.replace(/_/g, ' ')}
              </p>
            )}
          </div>
          <button
            onClick={() => {
              console.log('Run Analysis button clicked')
              setShowAnalysisDialog(true)
            }}
            disabled={runAnalysis.isPending}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            <Play className="h-4 w-4" />
            {runAnalysis.isPending ? 'Running...' : 'Run Analysis'}
          </button>
          {runAnalysis.isPending && (
            <p className="text-sm text-muted-foreground">
              Analysis is running in the background. Results will appear here when complete.
            </p>
          )}
        </div>

        {assessments.length > 0 ? (
          <div className="space-y-6 overflow-y-auto max-h-[calc(100vh-250px)]">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
              <div className="rounded-lg border bg-card p-4">
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Assessments</h3>
                <p className="text-2xl font-bold">{assessments.length}</p>
              </div>
              <div className="rounded-lg border bg-card p-4">
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Average Safety Score</h3>
                {(() => {
                  const average = assessments.length > 0
                    ? assessments.reduce((sum, a) => sum + (a.scores?.safety ?? a.overall_score ?? 0), 0) / assessments.length
                    : 0
                  // Same thresholds as SafetyScoreCard, so one score never shows two colours.
                  const color = average >= 0.8 ? 'text-green-600' : average >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                  return <p className={`text-2xl font-bold ${color}`}>{(average * 100).toFixed(1)}%</p>
                })()}
              </div>
              <div className="rounded-lg border bg-card p-4">
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Test Results</h3>
                <p className="text-2xl font-bold">{testResults.length}</p>
              </div>
              <div className="rounded-lg border bg-card p-4">
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Total Flags</h3>
                <p className="text-2xl font-bold text-red-600">
                  {flagsData.reduce((sum, f) => sum + f.value, 0)}
                </p>
              </div>
            </div>

            {/* Safety Score Cards */}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {assessments.map((assessment) => (
                <SafetyScoreCard key={assessment.id} assessment={assessment} />
              ))}
            </div>

            {/* Charts Row 1 */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {chartData.length > 0 && (
                <div className="rounded-lg border bg-card p-6">
                  <h2 className="text-lg font-semibold mb-4">Score Breakdown</h2>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Bar dataKey="value" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {scoreTrendData.length > 1 && (
                <div className="rounded-lg border bg-card p-6">
                  <h2 className="text-lg font-semibold mb-4">Safety Score Trend</h2>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={scoreTrendData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Line type="monotone" dataKey="safetyScore" stroke="#3b82f6" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Charts Row 2 */}
            {testResultsData.some(r => r.count > 0) && (
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div className="rounded-lg border bg-card p-6">
                  <h2 className="text-lg font-semibold mb-4">Test Results Score Distribution</h2>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={testResultsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {categoryData.length > 0 && (
                  <div className="rounded-lg border bg-card p-6">
                    <h2 className="text-lg font-semibold mb-4">Average Score by Category</h2>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={categoryData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Bar dataKey="avgScore" fill="#f59e0b" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            {/* Charts Row 3 */}
            {flagsData.length > 0 && (
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div className="rounded-lg border bg-card p-6">
                  <h2 className="text-lg font-semibold mb-4">Safety Flags Distribution</h2>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={flagsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#ef4444" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {multiAssessmentData.length > 0 && assessments.length > 1 && (
                  <div className="rounded-lg border bg-card p-6">
                    <h2 className="text-lg font-semibold mb-4">Score Comparison Across Assessments</h2>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={multiAssessmentData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Legend />
                        {assessments.map((_, idx) => (
                          <Bar 
                            key={idx} 
                            dataKey={`Assessment ${idx + 1}`} 
                            fill={COLORS[idx % COLORS.length]} 
                          />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            {/* Radar Chart for Multi-dimensional Analysis */}
            {chartData.length > 0 && (
              <div className="rounded-lg border bg-card p-6">
                <h2 className="text-lg font-semibold mb-4">Multi-Dimensional Score Analysis</h2>
                <ResponsiveContainer width="100%" height={400}>
                  <RadarChart data={chartData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="name" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Scores"
                      dataKey="value"
                      stroke="#3b82f6"
                      fill="#3b82f6"
                      fillOpacity={0.6}
                    />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}

            {assessments.map((assessment) => (
              <div key={assessment.id} className="rounded-lg border bg-card p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">Assessment Details</h3>
                  <span className="text-sm text-muted-foreground">
                    {formatDate(assessment.created_at)}
                  </span>
                </div>
                {assessment.assessment_text && (
                  <div className="rounded bg-muted/50 p-4">
                    <p className="text-sm whitespace-pre-wrap">{assessment.assessment_text}</p>
                  </div>
                )}
                {assessment.flags && assessment.flags.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Safety Flags</h4>
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
                  <div>
                    <h4 className="font-medium mb-2">Chain of Thought Analysis</h4>
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
                  <div>
                    <h4 className="font-medium mb-2">Factuality Analysis</h4>
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
                  <div>
                    <h4 className="font-medium mb-2">Manipulation Resistance Analysis</h4>
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
                  <div>
                    <h4 className="font-medium mb-2">Manipulation Capability Analysis</h4>
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
          <div className="rounded-lg border bg-card p-12 text-center">
            <p className="text-muted-foreground mb-4">No analysis available yet.</p>
            <button
              onClick={() => setShowAnalysisDialog(true)}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              Run Analysis
            </button>
          </div>
        )}
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
