import { Layout } from '@/components/layout/Layout'
import { MetricCard } from '@/components/common/MetricCard'
import { DataTable } from '@/components/common/DataTable'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { useTestRuns, useMultipleAssessments } from '@/lib/hooks'
import { TestRun, Assessment } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import { useMemo } from 'react'
import Link from 'next/link'
import { PlayCircle, TestTube, TrendingUp, Shield, AlertTriangle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend } from 'recharts'

export default function Dashboard() {
  const { data: testRuns = [], isLoading, error } = useTestRuns({ limit: 100 })

  // Get completed test run IDs for fetching assessments
  const completedRunIds = useMemo(() => 
    testRuns
      .filter(run => run.status === 'completed')
      .map(run => run.id),
    [testRuns]
  )

  const { data: assessmentsMap = new Map(), isLoading: loadingAssessments } = useMultipleAssessments(completedRunIds)

  const metrics = useMemo(() => {
    const total = testRuns.length
    const running = testRuns.filter((r) => r.status === 'running').length
    const completed = testRuns.filter((r) => r.status === 'completed').length
    const failed = testRuns.filter((r) => r.status === 'failed').length

    // Analysis metrics
    const allAssessments: Assessment[] = []
    assessmentsMap.forEach((assessments) => {
      allAssessments.push(...assessments)
    })
    
    const testsWithAnalysis = testRuns.filter(run => {
      const assessments = assessmentsMap.get(run.id) || []
      return assessments.length > 0
    }).length

    const avgOverallScore = allAssessments.length > 0
      ? allAssessments.reduce((sum, a) => sum + (a.overall_score || 0), 0) / allAssessments.length
      : 0

    const avgSafetyScore = allAssessments.length > 0
      ? allAssessments.reduce((sum, a) => sum + (a.scores?.safety || a.overall_score || 0), 0) / allAssessments.length
      : 0

    // Count COT detections
    const cotDetected = allAssessments.filter(a => 
      a.metadata?.cot_analysis?.cot_detected
    ).length

    // Count factuality checks
    const factualityChecks = allAssessments.filter(a => 
      a.metadata?.factuality_analysis
    ).length

    // Count manipulation analysis
    const manipulationAnalysis = allAssessments.filter(a => 
      a.metadata?.manipulation_resistance || a.metadata?.manipulation_capability
    ).length

    // Count flags/concerns
    const totalFlags = allAssessments.reduce((sum, a) => 
      sum + (a.flags?.length || 0), 0
    )

    return { 
      total, 
      running, 
      completed, 
      failed,
      testsWithAnalysis,
      totalAssessments: allAssessments.length,
      avgOverallScore,
      avgSafetyScore,
      cotDetected,
      factualityChecks,
      manipulationAnalysis,
      totalFlags,
    }
  }, [testRuns, assessmentsMap])

  const statusData = [
    { name: 'Completed', value: metrics.completed, color: '#10b981' },
    { name: 'Running', value: metrics.running, color: '#3b82f6' },
    { name: 'Pending', value: testRuns.filter((r) => r.status === 'pending').length, color: '#f59e0b' },
    { name: 'Failed', value: metrics.failed, color: '#ef4444' },
  ]

  const recentRuns = testRuns.slice(0, 10)

  // Get recent assessments with test run info
  const recentAssessments = useMemo(() => {
    const assessments: Array<{ assessment: Assessment; testRun?: TestRun }> = []
    assessmentsMap.forEach((assessmentsList, testRunId) => {
      const testRun = testRuns.find(r => r.id === testRunId)
      assessmentsList.forEach(assessment => {
        assessments.push({ assessment, testRun })
      })
    })
    return assessments
      .sort((a, b) => {
        const aDate = a.testRun?.created_at || ''
        const bDate = b.testRun?.created_at || ''
        return new Date(bDate).getTime() - new Date(aDate).getTime()
      })
      .slice(0, 5)
  }, [assessmentsMap, testRuns])

  // Score trends over time (last 10 assessments)
  const scoreTrendData = useMemo(() => {
    const allAssessments: Array<{ assessment: Assessment; testRun?: TestRun }> = []
    assessmentsMap.forEach((assessmentsList, testRunId) => {
      const testRun = testRuns.find(r => r.id === testRunId)
      assessmentsList.forEach(assessment => {
        allAssessments.push({ assessment, testRun })
      })
    })
    
    return allAssessments
      .sort((a, b) => {
        const aDate = a.testRun?.created_at || ''
        const bDate = b.testRun?.created_at || ''
        return new Date(aDate).getTime() - new Date(bDate).getTime()
      })
      .slice(-10)
      .map((item, index) => ({
        index: index + 1,
        overall: (item.assessment.overall_score || 0) * 100,
        safety: ((item.assessment.scores?.safety || item.assessment.overall_score || 0)) * 100,
        alignment: ((item.assessment.scores?.alignment || 0)) * 100,
        factuality: ((item.assessment.scores?.factuality || item.assessment.metadata?.factuality_analysis?.factuality_score || 0)) * 100,
      }))
  }, [assessmentsMap, testRuns])

  // Top models by average score
  const topModels = useMemo(() => {
    const modelScores: Record<string, { model: string; provider: string; scores: number[]; count: number }> = {}
    
    testRuns.forEach(run => {
      if (run.status !== 'completed') return
      const assessments = assessmentsMap.get(run.id) || []
      if (assessments.length === 0) return
      
      const key = `${run.patient_provider}:${run.patient_model}`
      if (!modelScores[key]) {
        modelScores[key] = {
          model: run.patient_model,
          provider: run.patient_provider,
          scores: [],
          count: 0,
        }
      }
      
      assessments.forEach(a => {
        if (a.overall_score) {
          modelScores[key].scores.push(a.overall_score)
          modelScores[key].count++
        }
      })
    })

    return Object.values(modelScores)
      .map(stat => ({
        ...stat,
        avgScore: stat.scores.length > 0 
          ? stat.scores.reduce((a, b) => a + b, 0) / stat.scores.length 
          : 0,
      }))
      .filter(m => m.avgScore > 0)
      .sort((a, b) => b.avgScore - a.avgScore)
      .slice(0, 5)
  }, [testRuns, assessmentsMap])

  const tableColumns = [
    {
      key: 'id',
      header: 'ID',
    },
    {
      key: 'test_type',
      header: 'Type',
      render: (run: TestRun) => (
        <span className="capitalize">{run.test_type}</span>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (run: TestRun) => <StatusBadge status={run.status} />,
    },
    {
      key: 'doctor_model',
      header: 'Doctor',
      render: (run: TestRun) => `${run.doctor_provider}/${run.doctor_model}`,
    },
    {
      key: 'patient_model',
      header: 'Patient',
      render: (run: TestRun) => `${run.patient_provider}/${run.patient_model}`,
    },
    {
      key: 'created_at',
      header: 'Created',
      render: (run: TestRun) => (run.created_at ? formatDate(run.created_at) : '-'),
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (run: TestRun) => (
        <Link
          href={`/test-runs/${run.id}`}
          className="text-primary hover:underline"
        >
          View
        </Link>
      ),
    },
  ]

  if (isLoading) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </Layout>
    )
  }

  if (error) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-2">Error loading data</h2>
            <p className="text-muted-foreground">
              {error instanceof Error ? error.message : 'An unexpected error occurred'}
            </p>
          </div>
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <div className="flex gap-3">
            <Link
              href="/create-test"
              className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              <PlayCircle className="h-4 w-4" />
              Create Test
            </Link>
            <Link
              href="/benchmarks"
              className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              <TestTube className="h-4 w-4" />
              Run Benchmark
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard title="Total Tests" value={metrics.total} />
          <MetricCard title="Running" value={metrics.running} />
          <MetricCard title="Completed" value={metrics.completed} />
          <MetricCard title="Failed" value={metrics.failed} />
        </div>

        {/* Analysis Metrics */}
        {metrics.totalAssessments > 0 && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5">
            <MetricCard 
              title="Tests with Analysis" 
              value={metrics.testsWithAnalysis}
              subtitle={`${metrics.completed > 0 ? ((metrics.testsWithAnalysis / metrics.completed) * 100).toFixed(0) : 0}% of completed`}
              icon={<TrendingUp className="h-5 w-5" />}
            />
            <MetricCard 
              title="Total Assessments" 
              value={metrics.totalAssessments}
              subtitle={`${metrics.avgOverallScore > 0 ? (metrics.avgOverallScore * 100).toFixed(1) : 0}% avg score`}
            />
            <MetricCard 
              title="Avg Safety Score" 
              value={`${(metrics.avgSafetyScore * 100).toFixed(1)}%`}
              icon={<Shield className="h-5 w-5" />}
            />
            <MetricCard 
              title="COT Detected" 
              value={metrics.cotDetected}
              subtitle={`${metrics.factualityChecks > 0 ? metrics.factualityChecks : 0} factuality checks`}
            />
            <MetricCard 
              title="Total Flags" 
              value={metrics.totalFlags}
              icon={<AlertTriangle className="h-5 w-5 text-red-600" />}
            />
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border bg-card p-6">
            <h2 className="mb-4 text-lg font-semibold">Status Overview</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-lg border bg-card p-6">
            <h2 className="mb-4 text-lg font-semibold">Recent Test Runs</h2>
            <DataTable
              data={recentRuns}
              columns={tableColumns}
              onRowClick={(run) => window.location.href = `/test-runs/${run.id}`}
              emptyMessage="No test runs yet"
            />
          </div>
        </div>

        {/* Analysis Section */}
        {metrics.totalAssessments > 0 && (
          <>
            {/* Score Trends */}
            {scoreTrendData.length > 0 && (
              <div className="rounded-lg border bg-card p-6">
                <h2 className="mb-4 text-lg font-semibold">Score Trends (Last 10 Assessments)</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={scoreTrendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="index" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                    <Legend />
                    <Line type="monotone" dataKey="overall" stroke="#3b82f6" strokeWidth={2} name="Overall" />
                    <Line type="monotone" dataKey="safety" stroke="#10b981" strokeWidth={2} name="Safety" />
                    <Line type="monotone" dataKey="alignment" stroke="#f59e0b" strokeWidth={2} name="Alignment" />
                    <Line type="monotone" dataKey="factuality" stroke="#ec4899" strokeWidth={2} name="Factuality" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Top Models and Recent Assessments */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Top Models */}
              {topModels.length > 0 && (
                <div className="rounded-lg border bg-card p-6">
                  <h2 className="mb-4 text-lg font-semibold">Top Performing Models</h2>
                  <div className="space-y-3">
                    {topModels.map((model, index) => (
                      <div key={`${model.provider}:${model.model}`} className="flex items-center justify-between rounded-lg border p-3">
                        <div className="flex items-center gap-3">
                          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-sm font-bold text-primary">
                            {index + 1}
                          </div>
                          <div>
                            <p className="font-medium">{model.model}</p>
                            <p className="text-xs text-muted-foreground">{model.provider}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-bold">{(model.avgScore * 100).toFixed(1)}%</p>
                          <p className="text-xs text-muted-foreground">{model.count} assessment{model.count !== 1 ? 's' : ''}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                  <Link 
                    href="/compare" 
                    className="mt-4 block text-center text-sm text-primary hover:underline"
                  >
                    View all models →
                  </Link>
                </div>
              )}

              {/* Recent Assessments */}
              {recentAssessments.length > 0 && (
                <div className="rounded-lg border bg-card p-6">
                  <h2 className="mb-4 text-lg font-semibold">Recent Assessments</h2>
                  <div className="space-y-3">
                    {recentAssessments.map(({ assessment, testRun }) => (
                      <Link
                        key={assessment.id}
                        href={`/test-runs/${assessment.test_run_id}`}
                        className="block rounded-lg border p-3 hover:bg-muted transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <p className="text-sm font-medium">
                              {testRun 
                                ? `${testRun.patient_provider}/${testRun.patient_model}`
                                : `Test Run #${assessment.test_run_id}`
                              }
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {testRun?.test_type ? testRun.test_type.charAt(0).toUpperCase() + testRun.test_type.slice(1) : 'Test'}
                              {testRun?.created_at && ` • ${formatDate(testRun.created_at)}`}
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="text-lg font-bold">{(assessment.overall_score * 100).toFixed(1)}%</p>
                            {assessment.scores?.safety && (
                              <p className="text-xs text-muted-foreground">
                                Safety: {(assessment.scores.safety * 100).toFixed(0)}%
                              </p>
                            )}
                          </div>
                        </div>
                        {assessment.flags && assessment.flags.length > 0 && (
                          <div className="mt-2 flex gap-1">
                            {assessment.flags.slice(0, 3).map((flag, idx) => (
                              <span key={idx} className="text-xs rounded bg-red-100 px-2 py-0.5 text-red-800">
                                {flag}
                              </span>
                            ))}
                            {assessment.flags.length > 3 && (
                              <span className="text-xs text-muted-foreground">+{assessment.flags.length - 3} more</span>
                            )}
                          </div>
                        )}
                      </Link>
                    ))}
                  </div>
                  <Link 
                    href="/test-runs" 
                    className="mt-4 block text-center text-sm text-primary hover:underline"
                  >
                    View all test runs →
                  </Link>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </Layout>
  )
}
