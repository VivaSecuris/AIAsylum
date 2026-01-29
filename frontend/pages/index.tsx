import { Layout } from '@/components/layout/Layout'
import { MetricCard } from '@/components/common/MetricCard'
import { DataTable } from '@/components/common/DataTable'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { useTestRuns, useMultipleAssessments, useSuites, useUpdateTestRun, useUpdateSuite } from '@/lib/hooks'
import { useQueryClient } from '@tanstack/react-query'
import { TestRun, Assessment, Suite } from '@/lib/api'
import { formatDate, formatDateTime } from '@/lib/utils'
import { useMemo, useState, useEffect } from 'react'
import Link from 'next/link'
import { PlayCircle, TestTube, TrendingUp, Shield, AlertTriangle, Layers, BarChart3, Edit2, Check, X } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend } from 'recharts'
import { toast } from '@/lib/toast'

// Inline editing component for renaming
function InlineEditableName({ 
  value, 
  onSave, 
  displayValue,
  className = ""
}: { 
  value: string
  onSave: (newValue: string) => void
  displayValue?: string
  className?: string
}) {
  const [isEditing, setIsEditing] = useState(false)
  const [name, setName] = useState(value)
  const [savedName, setSavedName] = useState<string | null>(null)

  useEffect(() => {
    // Only update from prop if we're not editing and haven't just saved
    if (!isEditing) {
      // If we just saved a name, use that until the prop updates
      if (savedName !== null && value === '') {
        // Keep the saved name if prop hasn't updated yet
        return
      }
      // Update from prop value
      setName(value || '')
      // Clear saved name once prop has updated
      if (savedName !== null && value === savedName) {
        setSavedName(null)
      }
    }
  }, [value, isEditing, savedName])

  const handleSave = () => {
    const nameToSave = name.trim()
    setSavedName(nameToSave) // Remember what we saved
    onSave(nameToSave)
    setIsEditing(false)
  }

  const handleCancel = () => {
    setName(value || '')
    setSavedName(null)
    setIsEditing(false)
  }

  if (isEditing) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          autoFocus
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              handleSave()
            } else if (e.key === 'Escape') {
              handleCancel()
            }
          }}
        />
        <button
          onClick={handleSave}
          className="p-1 text-green-600 hover:text-green-800"
          title="Save"
        >
          <Check className="h-4 w-4" />
        </button>
        <button
          onClick={handleCancel}
          className="p-1 text-red-600 hover:text-red-800"
          title="Cancel"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    )
  }

  // Use saved name if available, otherwise use displayValue or value
  const displayText = savedName !== null ? savedName : (displayValue || value || 'Unnamed')

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className="flex-1">{displayText}</span>
      <button
        onClick={() => setIsEditing(true)}
        className="p-1 text-muted-foreground hover:text-foreground"
        title="Rename"
      >
        <Edit2 className="h-4 w-4" />
      </button>
    </div>
  )
}

export default function Dashboard() {
  const { data: testRuns = [], isLoading, error } = useTestRuns({ limit: 100 })
  const { data: suites = [], isLoading: loadingSuites } = useSuites({ limit: 20 })
  const updateTestRun = useUpdateTestRun()
  const updateSuite = useUpdateSuite()
  const queryClient = useQueryClient()

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
      key: 'name',
      header: 'Name',
      render: (run: TestRun) => {
        const displayName = run.suite_name || run.meta_data?.name || `Test Run #${run.id}`
        return (
          <InlineEditableName
            value={run.meta_data?.name || ''}
            displayValue={displayName}
            onSave={(newName) => {
              updateTestRun.mutate(
                { id: run.id, data: { name: newName } },
                {
                  onSuccess: () => {
                    toast.success('Test run renamed successfully')
                  },
                  onError: (error) => {
                    console.error('Failed to rename test run:', error)
                    toast.error('Failed to rename test run')
                  },
                }
              )
            }}
          />
        )
      },
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
      header: 'Date & Time',
      render: (run: TestRun) => (
        <div className="text-sm">
          {run.created_at ? formatDateTime(run.created_at) : '-'}
        </div>
      ),
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
    const errorMessage = error instanceof Error ? error.message : String(error)
    const isNetworkError = errorMessage.includes('connect') || errorMessage.includes('Network Error')
    
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center max-w-md">
            <h2 className="text-xl font-semibold mb-2">Error loading data</h2>
            <p className="text-muted-foreground mb-4">
              {errorMessage}
            </p>
            {isNetworkError && (
              <div className="text-sm text-muted-foreground space-y-2">
                <p>Please check:</p>
                <ul className="list-disc list-inside space-y-1 text-left">
                  <li>The backend API server is running</li>
                  <li>The API URL is correct (check browser console)</li>
                  <li>No firewall is blocking the connection</li>
                </ul>
              </div>
            )}
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
              href="/suite"
              className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              <Layers className="h-4 w-4" />
              Create Suite
            </Link>
            <Link
              href="/create-test?type=benchmark"
              className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              <TestTube className="h-4 w-4" />
              Run Benchmark
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5">
          <MetricCard title="Total Tests" value={metrics.total} />
          <MetricCard title="Running" value={metrics.running} />
          <MetricCard title="Completed" value={metrics.completed} />
          <MetricCard title="Failed" value={metrics.failed} />
          <MetricCard 
            title="Test Suites" 
            value={suites.length} 
            subtitle={suites.filter(s => s.status === 'running').length > 0 ? `${suites.filter(s => s.status === 'running').length} running` : undefined}
            icon={<Layers className="h-5 w-5" />}
          />
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

        {/* First-Level Runnable Objects: Suites, Benchmarks, Analysis */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Suites Section */}
          <div className="rounded-lg border bg-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Test Suites
              </h2>
              <Link
                href="/suite"
                className="text-sm text-primary hover:underline"
              >
                View All
              </Link>
            </div>
            {loadingSuites ? (
              <LoadingSpinner size="sm" />
            ) : suites.length > 0 ? (
              <div className="space-y-3">
                {suites.slice(0, 5).map((suite) => (
                  <div
                    key={suite.id}
                    className="block rounded-lg border p-3 hover:bg-muted transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex-1">
                        <InlineEditableName
                          value={suite.name || ''}
                          displayValue={suite.name || `Unnamed Suite`}
                          onSave={(newName) => {
                            updateSuite.mutate(
                              { id: suite.id, data: { name: newName } },
                              {
                                onSuccess: () => {
                                  toast.success('Suite renamed successfully')
                                },
                                onError: () => {
                                  toast.error('Failed to rename suite')
                                },
                              }
                            )
                          }}
                        />
                        <p className="text-xs text-muted-foreground mt-1">
                          {suite.total_runs} test runs • {formatDate(suite.created_at)}
                        </p>
                      </div>
                      <StatusBadge status={suite.status} />
                    </div>
                    <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
                      <span>✓ {suite.completed_runs}</span>
                      {suite.running_runs > 0 && <span>⟳ {suite.running_runs}</span>}
                      {suite.failed_runs > 0 && <span className="text-red-600">✗ {suite.failed_runs}</span>}
                    </div>
                    <Link
                      href={`/suite/${suite.id}`}
                      className="mt-2 block text-xs text-primary hover:underline"
                    >
                      View Suite →
                    </Link>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p className="text-sm mb-3">No suites yet</p>
                <Link
                  href="/suite"
                  className="text-sm text-primary hover:underline"
                >
                  Create your first suite →
                </Link>
              </div>
            )}
          </div>

          {/* Benchmarks Section */}
          <div className="rounded-lg border bg-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <TestTube className="h-5 w-5" />
                Benchmarks
              </h2>
              <Link
                href="/create-test?type=benchmark"
                className="text-sm text-primary hover:underline"
              >
                Run New
              </Link>
            </div>
            <div className="space-y-3">
              {testRuns
                .filter(run => run.test_type === 'benchmark')
                .slice(0, 5)
                .map((run) => (
                  <div
                    key={run.id}
                    className="block rounded-lg border p-3 hover:bg-muted transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex-1">
                        <InlineEditableName
                          value={run.meta_data?.name || ''}
                          displayValue={run.meta_data?.name || run.meta_data?.benchmark || 'Benchmark'}
                          onSave={(newName) => {
                            updateTestRun.mutate(
                              { id: run.id, data: { name: newName } },
                              {
                                onSuccess: () => {
                                  toast.success('Benchmark renamed successfully')
                                },
                                onError: () => {
                                  toast.error('Failed to rename benchmark')
                                },
                              }
                            )
                          }}
                        />
                        <p className="text-xs text-muted-foreground mt-1">
                          {run.patient_provider}/{run.patient_model} • {formatDate(run.created_at)}
                        </p>
                      </div>
                      <StatusBadge status={run.status} />
                    </div>
                    <Link
                      href={`/test-runs/${run.id}`}
                      className="block text-xs text-primary hover:underline"
                    >
                      View Benchmark →
                    </Link>
                  </div>
                ))}
              {testRuns.filter(run => run.test_type === 'benchmark').length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <p className="text-sm mb-3">No benchmarks yet</p>
                  <Link
                    href="/create-test?type=benchmark"
                    className="text-sm text-primary hover:underline"
                  >
                    Run your first benchmark →
                  </Link>
                </div>
              )}
            </div>
          </div>

          {/* Analysis Section */}
          <div className="rounded-lg border bg-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Analysis
              </h2>
              <Link
                href="/test-runs"
                className="text-sm text-primary hover:underline"
              >
                View All
              </Link>
            </div>
            {recentAssessments.length > 0 ? (
              <div className="space-y-3">
                {recentAssessments.slice(0, 5).map(({ assessment, testRun }) => {
                  const displayName = testRun?.suite_name || testRun?.meta_data?.name || 
                    (testRun 
                      ? `${testRun.patient_provider}/${testRun.patient_model}`
                      : `Test Run #${assessment.test_run_id}`)
                  return (
                    <div
                      key={assessment.id}
                      className="block rounded-lg border p-3 hover:bg-muted transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex-1">
                          {testRun ? (
                            <InlineEditableName
                              value={testRun.meta_data?.name || ''}
                              displayValue={displayName}
                              onSave={(newName) => {
                                updateTestRun.mutate(
                                  { id: testRun.id, data: { name: newName } },
                                  {
                                    onSuccess: () => {
                                      toast.success('Analysis renamed successfully')
                                    },
                                    onError: () => {
                                      toast.error('Failed to rename analysis')
                                    },
                                  }
                                )
                              }}
                            />
                          ) : (
                            <p className="text-sm font-medium">{displayName}</p>
                          )}
                          <p className="text-xs text-muted-foreground mt-1">
                            {formatDate(testRun?.created_at || '')}
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-bold">{(assessment.overall_score * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                      <Link
                        href={`/analysis/${assessment.test_run_id}`}
                        className="block text-xs text-primary hover:underline"
                      >
                        View Analysis →
                      </Link>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p className="text-sm mb-3">No analyses yet</p>
                <p className="text-xs">Run tests and analysis will appear here</p>
              </div>
            )}
          </div>
        </div>

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
                              {testRun?.created_at && ` • ${formatDateTime(testRun.created_at)}`}
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
