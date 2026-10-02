import { useEffect, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useRouter } from 'next/router'
import Link from 'next/link'
import { Assessment, Suite, TestRun } from '@/lib/api'
import {
  hasActiveWork, LIVE_REFETCH_INTERVAL, useMultipleAssessments, useSuites,
  useTestRuns, useUpdateSuite, useUpdateTestRun,
} from '@/lib/hooks'
import {
  benchmarkName,
  shortModelName,
  useBenchmarkCampaigns,
  type BenchmarkCampaign,
} from '@/lib/benchmark-campaigns'
import { formatDateTime, parseApiDate } from '@/lib/utils'
import { testRunModelLabel, testRunModelRef } from '@/lib/test-run-display'
import { toast } from '@/lib/toast'
import { Column } from '@/components/common/DataTable'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { DashboardWidgetData, InlineEditableName } from './DashboardWidgets'

const EMPTY_RUNS: TestRun[] = []
const EMPTY_SUITES: Suite[] = []
const EMPTY_CAMPAIGNS: BenchmarkCampaign[] = []

const PROVIDER_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#14b8a6', '#ef4444', '#64748b']

/** Pick the newest campaign with scores; prefer completed campaigns. */
export function buildBenchmarkAccuracyRows(
  campaigns: BenchmarkCampaign[]
): Array<{ name: string; score: number }> {
  const withScores = campaigns.filter((campaign) =>
    campaign.runs.some((run) => typeof run.score === 'number' && Number.isFinite(run.score))
  )
  if (withScores.length === 0) return []
  const completed = withScores.filter((campaign) => campaign.status === 'completed')
  const pool = completed.length > 0 ? completed : withScores
  const newest = [...pool].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))[0]
  return newest.runs
    .filter((run) => typeof run.score === 'number' && Number.isFinite(run.score))
    .map((run) => ({
      name: `${shortModelName(run.model)} · ${benchmarkName(run.benchmark)}`,
      score: (run.score as number) * 100,
    }))
}

/** Shared, deterministic data for every widget, including valid zero scores. */
export function summarizeDashboardData(testRuns: TestRun[], assessmentsMap: Map<number, Assessment[]>) {
  const runsById = new Map(testRuns.map((run) => [run.id, run]))
  const assessments = Array.from(assessmentsMap.entries()).flatMap(([id, values]) =>
    values.map((assessment) => ({ assessment, testRun: runsById.get(id) }))
  )
  const timestamp = ({ testRun }: { testRun?: TestRun }) =>
    testRun?.created_at ? parseApiDate(testRun.created_at).getTime() || 0 : 0
  assessments.sort((a, b) => timestamp(a) - timestamp(b) || a.assessment.id - b.assessment.id)
  const allAssessments = assessments.map(({ assessment }) => assessment)
  const eligible = testRuns.filter((run) => run.status === 'completed' && !['benchmark', 'analysis'].includes(run.test_type))
  const average = (score: (assessment: Assessment) => number) => allAssessments.length
    ? allAssessments.reduce((sum, item) => sum + score(item), 0) / allAssessments.length : 0
  const avgOverallScore = average((item) => item.overall_score ?? 0)
  const avgSafetyScore = average((item) => item.scores?.safety ?? item.overall_score ?? 0)
  const avgAlignmentScore = average((item) => item.scores?.alignment ?? 0)
  const avgFactualityScore = average((item) =>
    item.scores?.factuality ?? item.metadata?.factuality_analysis?.factuality_score ?? 0
  )
  const metrics = {
    total: testRuns.length,
    running: testRuns.filter((run) => run.status === 'running').length,
    pending: testRuns.filter((run) => run.status === 'pending').length,
    completed: testRuns.filter((run) => run.status === 'completed').length,
    failed: testRuns.filter((run) => run.status === 'failed').length,
    testsWithAnalysis: eligible.filter((run) => (assessmentsMap.get(run.id)?.length ?? 0) > 0).length,
    analysisEligibleCompleted: eligible.length,
    totalAssessments: allAssessments.length,
    avgOverallScore,
    avgSafetyScore,
    cotDetected: allAssessments.filter((item) => item.metadata?.cot_analysis?.cot_detected).length,
    factualityChecks: allAssessments.filter((item) => item.metadata?.factuality_analysis).length,
    manipulationAnalysis: allAssessments.filter((item) => item.metadata?.manipulation_resistance || item.metadata?.manipulation_capability).length,
    totalFlags: allAssessments.reduce((sum, item) => sum + (item.flags?.length ?? 0), 0),
  }
  const typeCounts = new Map<string, number>()
  const providerCounts = new Map<string, number>()
  const modelScores = new Map<string, { model: string; provider: string; scores: number[] }>()
  testRuns.forEach((run) => {
    typeCounts.set(run.test_type, (typeCounts.get(run.test_type) ?? 0) + 1)
    const provider = run.patient_provider || 'unknown'
    providerCounts.set(provider, (providerCounts.get(provider) ?? 0) + 1)
    if (run.status !== 'completed') return
    const model = testRunModelRef(run)
    const key = JSON.stringify([run.patient_provider, model])
    for (const assessment of assessmentsMap.get(run.id) ?? []) {
      if (typeof assessment.overall_score !== 'number' || !Number.isFinite(assessment.overall_score)) continue
      const group = modelScores.get(key) ?? { model, provider: run.patient_provider, scores: [] }
      group.scores.push(assessment.overall_score)
      modelScores.set(key, group)
    }
  })
  const radarScores = allAssessments.length > 0
    ? [
        { subject: 'Overall', value: avgOverallScore * 100 },
        { subject: 'Safety', value: avgSafetyScore * 100 },
        { subject: 'Alignment', value: avgAlignmentScore * 100 },
        { subject: 'Factuality', value: avgFactualityScore * 100 },
      ]
    : []
  const providerMix = Array.from(providerCounts.entries()).map(([name, value], index) => ({
    name,
    value,
    color: PROVIDER_COLORS[index % PROVIDER_COLORS.length],
  }))
  return {
    metrics,
    statusData: [
      { name: 'Completed', value: metrics.completed, color: '#10b981' },
      { name: 'Running', value: metrics.running, color: '#3b82f6' },
      { name: 'Pending', value: metrics.pending, color: '#f59e0b' },
      { name: 'Failed', value: metrics.failed, color: '#ef4444' },
    ],
    typeBreakdown: Array.from(typeCounts).map(([name, count]) => ({ name: name.replace(/_/g, ' '), count })),
    providerMix,
    radarScores,
    recentRuns: testRuns.slice(0, 10),
    recentAssessments: assessments.slice(-5).reverse(),
    scoreTrendData: assessments.slice(-10).map(({ assessment }, index) => ({
      index: index + 1,
      overall: (assessment.overall_score ?? 0) * 100,
      safety: (assessment.scores?.safety ?? assessment.overall_score ?? 0) * 100,
      alignment: (assessment.scores?.alignment ?? 0) * 100,
      factuality: (assessment.scores?.factuality ?? assessment.metadata?.factuality_analysis?.factuality_score ?? 0) * 100,
    })),
    topModels: Array.from(modelScores.values()).map(({ model, provider, scores }) => ({
      model, provider, count: scores.length,
      avgScore: scores.reduce((sum, score) => sum + score, 0) / scores.length,
    })).sort((a, b) => b.avgScore - a.avgScore).slice(0, 5),
  }
}

export function useDashboardData() {
  const queryClient = useQueryClient()
  const router = useRouter()
  const [pollTogether, setPollTogether] = useState(false)
  const runsQuery = useTestRuns({ limit: 100 }, { live: true, active: pollTogether })
  const suitesQuery = useSuites({ limit: 20 }, { live: true, active: pollTogether })
  const campaignsQuery = useBenchmarkCampaigns()
  const testRuns = runsQuery.data ?? EMPTY_RUNS
  const suites = suitesQuery.data ?? EMPTY_SUITES
  const campaigns = campaignsQuery.data?.campaigns ?? EMPTY_CAMPAIGNS
  const isLive = hasActiveWork(testRuns) || hasActiveWork(suites)
  useEffect(() => { setPollTogether(isLive) }, [isLive])
  const completedIds = useMemo(() => testRuns.filter((run) => run.status === 'completed').map((run) => run.id), [testRuns])
  const assessmentsQuery = useMultipleAssessments(completedIds, {
    refetchInterval: isLive ? LIVE_REFETCH_INTERVAL : false,
  })
  const wasLive = useRef(false)
  useEffect(() => {
    // Capture the final analysis results even when completion stops the interval.
    if (wasLive.current && !isLive) {
      void queryClient.invalidateQueries({ queryKey: ['assessments'] })
    }
    wasLive.current = isLive
  }, [isLive, queryClient])
  const summary = useMemo(() => summarizeDashboardData(testRuns, assessmentsQuery.data), [testRuns, assessmentsQuery.data])
  const benchmarkAccuracy = useMemo(() => buildBenchmarkAccuracyRows(campaigns), [campaigns])
  const updateRun = useUpdateTestRun()
  const updateSuite = useUpdateSuite()
  const onRenameRun = (id: number, name: string) => updateRun.mutate({ id, data: { name } }, {
    onSuccess: () => toast.success('Test run renamed successfully'),
    onError: () => toast.error('Failed to rename test run'),
  })
  const onRenameSuite = (id: number, name: string) => updateSuite.mutate({ id, data: { name } }, {
    onSuccess: () => toast.success('Suite renamed successfully'),
    onError: () => toast.error('Failed to rename suite'),
  })
  const tableColumns: Column<TestRun>[] = [
    { key: 'id', header: 'ID' },
    { key: 'name', header: 'Name', render: (run) => (
      <div>
        <InlineEditableName value={run.meta_data?.name || ''} displayValue={run.meta_data?.name || `Test Run #${run.id}`}
          onSave={(name) => onRenameRun(run.id, name)} />
        {run.suite_name && <p className="text-xs text-muted-foreground">Suite: {run.suite_name}</p>}
      </div>
    ) },
    { key: 'test_type', header: 'Type', render: (run) => <span className="capitalize">{run.test_type.replace(/_/g, ' ')}</span> },
    { key: 'status', header: 'Status', render: (run) => <StatusBadge status={run.status} /> },
    { key: 'doctor_model', header: 'Doctor', render: (run) => run.test_type === 'benchmark' ? '—' : `${run.doctor_provider}/${run.doctor_model}` },
    { key: 'patient_model', header: 'Patient', render: testRunModelLabel },
    { key: 'created_at', header: 'Date & Time', render: (run) => run.created_at ? formatDateTime(run.created_at) : '—' },
    { key: 'actions', header: 'Actions', render: (run) => (
      <Link href={`/test-runs/${run.id}`} onClick={(event) => event.stopPropagation()} className="text-primary hover:underline">View</Link>
    ) },
  ]
  const data: DashboardWidgetData = {
    ...summary, benchmarkAccuracy, testRuns, suites, loadingSuites: suitesQuery.isLoading,
    tableColumns, onRenameRun, onRenameSuite,
    onRowClick: (run) => { void router.push(`/test-runs/${run.id}`) },
  }
  return {
    data, isLive,
    isLoading: runsQuery.isLoading,
    initialError: runsQuery.data === undefined ? runsQuery.error : null,
    refreshError: runsQuery.error || suitesQuery.error || assessmentsQuery.error || campaignsQuery.error,
    retry: () => {
      void runsQuery.refetch()
      void suitesQuery.refetch()
      void campaignsQuery.refetch()
      void queryClient.invalidateQueries({ queryKey: ['assessments'] })
    },
  }
}
