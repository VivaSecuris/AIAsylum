import Link from 'next/link'
import { useEffect, useState } from 'react'
import {
  AlertTriangle,
  BarChart3,
  Check,
  Edit2,
  Layers,
  Shield,
  TestTube,
  TrendingUp,
  X,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { MetricCard } from '@/components/common/MetricCard'
import { Column, DataTable } from '@/components/common/DataTable'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { BenchmarkResultsChart } from '@/components/charts/BenchmarkResultsChart'
import { Assessment, Suite, TestRun } from '@/lib/api'
import { formatDate, formatDateTime } from '@/lib/utils'
import { testRunModelLabel } from '@/lib/test-run-display'
import { DashboardWidgetId } from '@/lib/dashboard-layout'

export interface DashboardMetrics {
  total: number
  running: number
  completed: number
  failed: number
  pending: number
  testsWithAnalysis: number
  analysisEligibleCompleted: number
  totalAssessments: number
  avgOverallScore: number
  avgSafetyScore: number
  cotDetected: number
  factualityChecks: number
  manipulationAnalysis: number
  totalFlags: number
}

export interface TopModelStat {
  model: string
  provider: string
  avgScore: number
  count: number
}

export interface ScoreTrendPoint {
  index: number
  overall: number
  safety: number
  alignment: number
  factuality: number
}

export interface DashboardWidgetData {
  metrics: DashboardMetrics
  statusData: Array<{ name: string; value: number; color: string }>
  typeBreakdown: Array<{ name: string; count: number }>
  providerMix: Array<{ name: string; value: number; color: string }>
  radarScores: Array<{ subject: string; value: number }>
  benchmarkAccuracy: Array<{ name: string; score: number }>
  scoreTrendData: ScoreTrendPoint[]
  topModels: TopModelStat[]
  recentRuns: TestRun[]
  suites: Suite[]
  loadingSuites: boolean
  recentAssessments: Array<{ assessment: Assessment; testRun?: TestRun }>
  testRuns: TestRun[]
  tableColumns: Column<TestRun>[]
  onRowClick: (run: TestRun) => void
  onRenameSuite: (id: number, name: string) => void
  onRenameRun: (id: number, name: string) => void
}

function StatusPieChartInner({
  data,
  width = 300,
  height = 300,
}: {
  data: Array<{ name: string; value: number; color: string }>
  width?: number
  height?: number
}) {
  const size = Math.min(width, height)
  const outerRadius = Math.max(0, size / 2 - 8)
  const slices = data.filter((entry) => entry.value > 0)
  return (
    <PieChart width={width} height={height}>
      <Pie
        data={slices}
        cx="50%"
        cy="50%"
        labelLine={false}
        label={false}
        outerRadius={outerRadius}
        fill="#8884d8"
        dataKey="value"
      >
        {slices.map((entry) => (
          <Cell key={entry.name} fill={entry.color} />
        ))}
      </Pie>
      <Tooltip />
    </PieChart>
  )
}

export function InlineEditableName({
  value,
  onSave,
  displayValue,
  className = '',
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
    if (!isEditing) {
      if (savedName !== null && value === '') return
      setName(value || '')
      if (savedName !== null && value === savedName) setSavedName(null)
    }
  }, [value, isEditing, savedName])

  const stop = (e: React.SyntheticEvent) => e.stopPropagation()

  if (isEditing) {
    return (
      <div className={`flex items-center gap-2 ${className}`} onClick={stop} onMouseDown={stop}>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm"
          autoFocus
          onKeyDown={(e) => {
            e.stopPropagation()
            if (e.key === 'Enter') {
              e.preventDefault()
              setSavedName(name.trim())
              onSave(name.trim())
              setIsEditing(false)
            } else if (e.key === 'Escape') {
              setName(value || '')
              setSavedName(null)
              setIsEditing(false)
            }
          }}
        />
        <button
          type="button"
          onClick={() => {
            setSavedName(name.trim())
            onSave(name.trim())
            setIsEditing(false)
          }}
          className="p-1 text-green-600"
          title="Save"
        >
          <Check className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={() => {
            setName(value || '')
            setSavedName(null)
            setIsEditing(false)
          }}
          className="p-1 text-red-600"
          title="Cancel"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    )
  }

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className="flex-1">{savedName || displayValue || value || 'Unnamed'}</span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          setIsEditing(true)
        }}
        onMouseDown={stop}
        className="p-1 text-muted-foreground hover:text-foreground"
        title="Rename"
      >
        <Edit2 className="h-4 w-4" />
      </button>
    </div>
  )
}

export function renderDashboardWidget(id: DashboardWidgetId, data: DashboardWidgetData) {
  const { metrics } = data
  // Grid columns follow each resizable widget, independent of viewport breakpoints.
  const metricGrid = 'grid min-h-full grid-cols-[repeat(auto-fit,minmax(min(100%,9rem),1fr))] gap-3'

  switch (id) {
    case 'metrics':
      return (
        <div className={metricGrid}>
          <MetricCard className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0" title="Total Tests" value={metrics.total} />
          <MetricCard className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0" title="Running" value={metrics.running} />
          <MetricCard className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0" title="Completed" value={metrics.completed} />
          <MetricCard className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0" title="Failed" value={metrics.failed} />
          <MetricCard
            className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0"
            title="Test Suites"
            value={data.suites.length}
            subtitle={
              data.suites.filter((s) => s.status === 'running').length > 0
                ? `${data.suites.filter((s) => s.status === 'running').length} running`
                : undefined
            }
            icon={<Layers className="h-5 w-5" />}
          />
        </div>
      )

    case 'analysis-metrics':
      if (metrics.totalAssessments === 0) {
        return <p className="text-sm text-muted-foreground">No assessments yet. Run analysis on completed behavioral tests.</p>
      }
      return (
        <div className={metricGrid}>
          <MetricCard
            className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0"
            title="Behavioral Tests with Analysis"
            value={metrics.testsWithAnalysis}
            subtitle={`${metrics.analysisEligibleCompleted > 0 ? ((metrics.testsWithAnalysis / metrics.analysisEligibleCompleted) * 100).toFixed(0) : 0}% of completed behavioral tests`}
            icon={<TrendingUp className="h-5 w-5" />}
          />
          <MetricCard
            className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0"
            title="Total Assessments"
            value={metrics.totalAssessments}
            subtitle={`${metrics.avgOverallScore > 0 ? (metrics.avgOverallScore * 100).toFixed(1) : 0}% avg score`}
          />
          <MetricCard
            className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0"
            title="Avg Safety Score"
            value={`${(metrics.avgSafetyScore * 100).toFixed(1)}%`}
            icon={<Shield className="h-5 w-5" />}
          />
          <MetricCard
            className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0"
            title="COT Detected"
            value={metrics.cotDetected}
            subtitle={`${metrics.factualityChecks} factuality checks`}
          />
          <MetricCard
            className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0"
            title="Total Flags"
            value={metrics.totalFlags}
            icon={<AlertTriangle className="h-5 w-5 text-red-600" />}
          />
        </div>
      )

    case 'suites':
      return (
        <div className="flex h-full min-h-0 flex-col">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-2 font-semibold">
              <Layers className="h-4 w-4" /> Test Suites
            </h3>
            <Link href="/suite" className="text-sm text-primary hover:underline">View All</Link>
          </div>
          {data.loadingSuites ? (
            <LoadingSpinner size="sm" />
          ) : data.suites.length > 0 ? (
            <div className="min-h-0 space-y-3 overflow-auto">
              {data.suites.slice(0, 5).map((suite) => (
                <div key={suite.id} className="rounded-lg border p-3">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <InlineEditableName
                      value={suite.name || ''}
                      displayValue={suite.name || 'Unnamed Suite'}
                      onSave={(newName) => data.onRenameSuite(suite.id, newName)}
                    />
                    <StatusBadge status={suite.status} />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {suite.total_runs} test runs • {formatDate(suite.created_at)}
                  </p>
                  <Link href={`/suite/${suite.id}`} className="mt-2 block text-xs text-primary hover:underline">
                    View Suite →
                  </Link>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No suites yet</p>
          )}
        </div>
      )

    case 'benchmarks': {
      const benchmarks = data.testRuns.filter((run) => run.test_type === 'benchmark')
      return (
        <div className="flex h-full min-h-0 flex-col">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-2 font-semibold">
              <TestTube className="h-4 w-4" /> Benchmarks
            </h3>
            <Link href="/benchmarks" className="text-sm text-primary hover:underline">Run New</Link>
          </div>
          <div className="min-h-0 space-y-3 overflow-auto">
            {benchmarks.slice(0, 5).map((run) => (
              <div key={run.id} className="rounded-lg border p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <InlineEditableName
                    value={run.meta_data?.name || ''}
                    displayValue={run.meta_data?.name || run.meta_data?.benchmark || 'Benchmark'}
                    onSave={(newName) => data.onRenameRun(run.id, newName)}
                  />
                  <StatusBadge status={run.status} />
                </div>
                <p className="text-xs text-muted-foreground">
                  {testRunModelLabel(run)} • {formatDate(run.created_at)}
                </p>
                <Link href={`/test-runs/${run.id}`} className="mt-2 block text-xs text-primary hover:underline">
                  View Benchmark →
                </Link>
              </div>
            ))}
            {benchmarks.length === 0 && <p className="text-sm text-muted-foreground">No benchmarks yet</p>}
          </div>
        </div>
      )
    }

    case 'analysis-list':
      return (
        <div className="flex h-full min-h-0 flex-col">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-2 font-semibold">
              <BarChart3 className="h-4 w-4" /> Analysis
            </h3>
            <Link href="/test-runs" className="text-sm text-primary hover:underline">View All</Link>
          </div>
          {data.recentAssessments.length > 0 ? (
            <div className="min-h-0 space-y-3 overflow-auto">
              {data.recentAssessments.slice(0, 5).map(({ assessment, testRun }) => {
                const displayName =
                  testRun?.meta_data?.name ||
                  (testRun ? testRunModelLabel(testRun) : `Test Run #${assessment.test_run_id}`)
                return (
                  <div key={assessment.id} className="rounded-lg border p-3">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      {testRun ? (
                        <InlineEditableName
                          value={testRun.meta_data?.name || ''}
                          displayValue={displayName}
                          onSave={(newName) => data.onRenameRun(testRun.id, newName)}
                        />
                      ) : (
                        <p className="text-sm font-medium">{displayName}</p>
                      )}
                      <p className="text-lg font-bold">{(assessment.overall_score * 100).toFixed(1)}%</p>
                    </div>
                    <Link href={`/analysis/${assessment.test_run_id}`} className="text-xs text-primary hover:underline">
                      View Analysis →
                    </Link>
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No analyses yet</p>
          )}
        </div>
      )

    case 'status-pie':
      return (
        <div className="flex h-full min-h-[180px] flex-col">
          <h3 className="mb-2 font-semibold">Status Overview</h3>
          <div className="min-h-0 flex-1">
            {data.statusData.some((entry) => entry.value > 0) ? (
              <ResponsiveContainer width="100%" height="100%">
                <StatusPieChartInner data={data.statusData} />
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-sm text-muted-foreground">No test runs yet</p>
            )}
          </div>
          <ul aria-label="Run status counts" className="mt-2 flex shrink-0 flex-wrap justify-center gap-x-4 gap-y-1 text-xs">
            {data.statusData.map((entry) => (
              <li key={entry.name} className="inline-flex items-center gap-1.5 whitespace-nowrap">
                <span aria-hidden="true" className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: entry.color }} />
                <span>{entry.name}: <strong>{entry.value}</strong></span>
              </li>
            ))}
          </ul>
        </div>
      )

    case 'recent-runs':
      return (
        <div className="flex h-full min-h-0 flex-col overflow-auto">
          <h3 className="mb-3 font-semibold">Recent Test Runs</h3>
          <DataTable
            data={data.recentRuns}
            columns={data.tableColumns}
            onRowClick={data.onRowClick}
            emptyMessage="No test runs yet"
          />
        </div>
      )

    case 'score-trends':
      if (data.scoreTrendData.length === 0) {
        return <p className="text-sm text-muted-foreground">Score trends appear after assessments are recorded.</p>
      }
      return (
        <div className="flex h-full min-h-0 flex-col">
          <h3 className="mb-2 font-semibold">Score Trends (Last 10 Assessments)</h3>
          <div className="min-h-0 flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.scoreTrendData}>
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
        </div>
      )

    case 'top-models':
      if (data.topModels.length === 0) {
        return <p className="text-sm text-muted-foreground">Top models appear after scored assessments exist.</p>
      }
      return (
        <div className="flex h-full min-h-0 flex-col overflow-auto">
          <h3 className="mb-3 font-semibold">Top Performing Models</h3>
          <div className="space-y-3">
            {data.topModels.map((model, index) => (
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
                  <p className="text-xs text-muted-foreground">
                    {model.count} assessment{model.count !== 1 ? 's' : ''}
                  </p>
                </div>
              </div>
            ))}
          </div>
          <Link href="/compare" className="mt-4 block text-center text-sm text-primary hover:underline">
            View all models →
          </Link>
        </div>
      )

    case 'recent-assessments':
      if (data.recentAssessments.length === 0) {
        return <p className="text-sm text-muted-foreground">No assessments yet</p>
      }
      return (
        <div className="flex h-full min-h-0 flex-col overflow-auto">
          <h3 className="mb-3 font-semibold">Recent Assessments</h3>
          <div className="space-y-3">
            {data.recentAssessments.map(({ assessment, testRun }) => (
              <Link
                key={assessment.id}
                href={`/test-runs/${assessment.test_run_id}`}
                className="block rounded-lg border p-3 hover:bg-muted transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">
                      {testRun ? testRunModelLabel(testRun) : `Test Run #${assessment.test_run_id}`}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {testRun?.test_type
                        ? testRun.test_type.charAt(0).toUpperCase() + testRun.test_type.slice(1)
                        : 'Test'}
                      {testRun?.created_at && ` • ${formatDateTime(testRun.created_at)}`}
                    </p>
                  </div>
                  <p className="text-lg font-bold">{(assessment.overall_score * 100).toFixed(1)}%</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )

    case 'test-type-breakdown':
      return (
        <div className="flex h-full min-h-0 flex-col">
          <h3 className="mb-2 font-semibold">Test Type Breakdown</h3>
          {data.typeBreakdown.length === 0 ? (
            <p className="text-sm text-muted-foreground">No runs yet</p>
          ) : (
            <div className="min-h-0 flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.typeBreakdown}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" name="Runs" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )

    case 'flags-summary':
      return (
        <div className={`${metricGrid} content-start`}>
          <MetricCard className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0" title="COT Detected" value={metrics.cotDetected} />
          <MetricCard className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0" title="Factuality Checks" value={metrics.factualityChecks} />
          <MetricCard className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0" title="Manipulation Analysis" value={metrics.manipulationAnalysis} />
          <MetricCard
            className="min-w-0 break-words p-4 [&_.flex-1]:min-w-0"
            title="Total Flags"
            value={metrics.totalFlags}
            icon={<AlertTriangle className="h-5 w-5 text-red-600" />}
          />
        </div>
      )

    case 'safety-radar':
      if (!data.radarScores?.length) {
        return (
          <p className="text-sm text-muted-foreground">
            Safety radar appears after assessments are recorded.{' '}
            <Link href="/test-runs" className="text-primary hover:underline">Open test runs</Link>
          </p>
        )
      }
      return (
        <div className="flex h-full min-h-0 flex-col">
          <h3 className="mb-2 font-semibold">Safety Radar</h3>
          <div className="min-h-0 flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={data.radarScores}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis domain={[0, 100]} />
                <Radar name="Average" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.35} />
                <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )

    case 'provider-mix':
      return (
        <div className="flex h-full min-h-[180px] flex-col">
          <h3 className="mb-2 font-semibold">Provider Mix</h3>
          <div className="min-h-0 flex-1">
            {data.providerMix?.some((entry) => entry.value > 0) ? (
              <ResponsiveContainer width="100%" height="100%">
                <StatusPieChartInner data={data.providerMix} />
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-sm text-muted-foreground">
                No test runs yet.{' '}
                <Link href="/create-test" className="ml-1 text-primary hover:underline">Create a test</Link>
              </p>
            )}
          </div>
          <ul aria-label="Patient provider counts" className="mt-2 flex shrink-0 flex-wrap justify-center gap-x-4 gap-y-1 text-xs">
            {(data.providerMix ?? []).map((entry) => (
              <li key={entry.name} className="inline-flex items-center gap-1.5 whitespace-nowrap">
                <span aria-hidden="true" className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: entry.color }} />
                <span>{entry.name}: <strong>{entry.value}</strong></span>
              </li>
            ))}
          </ul>
        </div>
      )

    case 'suite-progress': {
      const rows = (data.suites ?? []).slice(0, 8).map((suite) => ({
        name: suite.name || `Suite #${suite.id}`,
        completed: suite.completed_runs ?? 0,
        running: suite.running_runs ?? 0,
        failed: suite.failed_runs ?? 0,
        pending: suite.pending_runs ?? 0,
      }))
      if (rows.length === 0) {
        return (
          <p className="text-sm text-muted-foreground">
            No suites yet.{' '}
            <Link href="/suite" className="text-primary hover:underline">Create a suite</Link>
          </p>
        )
      }
      return (
        <div className="flex h-full min-h-0 flex-col">
          <h3 className="mb-2 font-semibold">Suite Progress</h3>
          <div className="min-h-0 flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 12 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="name" width={96} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="completed" stackId="progress" fill="#10b981" name="Completed" />
                <Bar dataKey="running" stackId="progress" fill="#3b82f6" name="Running" />
                <Bar dataKey="pending" stackId="progress" fill="#f59e0b" name="Pending" />
                <Bar dataKey="failed" stackId="progress" fill="#ef4444" name="Failed" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )
    }

    case 'benchmark-accuracy':
      if (!data.benchmarkAccuracy?.length) {
        return (
          <p className="text-sm text-muted-foreground">
            Benchmark accuracy appears after a scored campaign finishes.{' '}
            <Link href="/benchmarks" className="text-primary hover:underline">Open benchmarks</Link>
          </p>
        )
      }
      return (
        <div className="flex h-full min-h-0 flex-col">
          <h3 className="mb-2 font-semibold">Benchmark Accuracy</h3>
          <div className="min-h-0 flex-1">
            <BenchmarkResultsChart data={data.benchmarkAccuracy} type="bar" height="100%" />
          </div>
        </div>
      )

    default:
      return null
  }
}
