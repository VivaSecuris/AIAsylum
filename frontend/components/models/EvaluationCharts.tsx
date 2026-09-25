import { useState, useMemo } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { DataTable } from '@/components/common/DataTable'
import { useTestRuns } from '@/lib/hooks'
import { apiClient, Assessment } from '@/lib/api'
import { testRunModelRef } from '@/lib/test-run-display'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, ComposedChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'

interface EvaluationChartsProps {
  /** Undefined shows all models; an empty list deliberately shows no models. */
  modelRefs?: string[]
  displayNames?: Record<string, string>
}

const formatLabel = (value: string) => value.replace(/_/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase())

export function EvaluationCharts({ modelRefs, displayNames = {} }: EvaluationChartsProps) {
  const { data: allTestRuns = [], isLoading, error: runsError, refetch: refetchRuns } = useTestRuns({ limit: 1000 })
  const testRuns = useMemo(() => modelRefs === undefined ? allTestRuns : allTestRuns.filter(run => modelRefs.includes(testRunModelRef(run))), [allTestRuns, modelRefs])
  const modelLabel = (model: string) => displayNames[model] || model
  const availableTestTypes = useMemo(() => Array.from(new Set([
    'conversation', 'scenario', 'adversarial', 'one_shot', 'multi_shot', 'benchmark', 'group_therapy',
    ...testRuns.map(run => run.test_type).filter(Boolean),
  ])).sort(), [testRuns])
  const [selectedTestType, setSelectedTestType] = useState<string>('')
  const [selectedMetric, setSelectedMetric] = useState<string>('overall_score')

  // Fetch assessments for all completed test runs
  const completedRunIds = useMemo(() => 
    testRuns
      .filter(run => run.status === 'completed' && (!selectedTestType || run.test_type === selectedTestType))
      .map(run => run.id),
    [testRuns, selectedTestType]
  )

  const { data: assessmentData, isLoading: loadingAssessments, error: assessmentsError, refetch: refetchAssessments } = useQuery({
    queryKey: ['evaluation-chart-assessments', completedRunIds],
    enabled: completedRunIds.length > 0,
    queryFn: async () => {
      const responses = await Promise.allSettled(completedRunIds.map(id => apiClient.getAssessments(id)))
      const assessments = new Map<number, Assessment[]>()
      const failedRunIds: number[] = []
      responses.forEach((response, index) => {
        if (response.status === 'fulfilled') assessments.set(completedRunIds[index], response.value)
        else failedRunIds.push(completedRunIds[index])
      })
      return { assessments, failedRunIds }
    },
  })
  const assessmentsMap = useMemo(() => assessmentData?.assessments || new Map<number, Assessment[]>(), [assessmentData])
  const failedAssessmentIds = assessmentData?.failedRunIds || []

  // Aggregate assessment data by model
  const modelStats = useMemo(() => {
    const stats: Record<string, {
      provider: string
      model: string
      testCount: number
      completedCount: number
      avgScore: number
      testTypes: Record<string, number>
      statusCounts: Record<string, number>
      scores: number[]
      assessments: Assessment[]
      // Analysis metrics
      avgOverallScore: number | null
      avgScores: Record<string, number>
      cotDetectedCount: number
      cotTotalCount: number
      avgFactualityScore: number
      factualityCount: number
      avgManipulationResistance: number
      manipulationResistanceCount: number
      avgManipulationCapability: number
      manipulationCapabilityCount: number
    }> = {}

    testRuns.forEach((run) => {
      if (selectedTestType && run.test_type !== selectedTestType) return

      const model = testRunModelRef(run)
      const key = `${run.patient_provider}:${model}`
      if (!stats[key]) {
        stats[key] = {
          provider: run.patient_provider,
          model,
          testCount: 0,
          completedCount: 0,
          avgScore: 0,
          testTypes: {},
          statusCounts: {},
          scores: [],
          assessments: [],
          avgOverallScore: null,
          avgScores: {},
          cotDetectedCount: 0,
          cotTotalCount: 0,
          avgFactualityScore: 0,
          factualityCount: 0,
          avgManipulationResistance: 0,
          manipulationResistanceCount: 0,
          avgManipulationCapability: 0,
          manipulationCapabilityCount: 0,
        }
      }
      stats[key].testCount++
      stats[key].statusCounts[run.status] = (stats[key].statusCounts[run.status] || 0) + 1
      if (run.status === 'completed') {
        stats[key].completedCount++
        // Get assessments for this test run
        const assessments = assessmentsMap.get(run.id) || []
        stats[key].assessments.push(...assessments)
      }
      if (run.test_type) {
        stats[key].testTypes[run.test_type] = (stats[key].testTypes[run.test_type] || 0) + 1
      }
    })

    // Calculate aggregated metrics from assessments
    Object.values(stats).forEach(stat => {
      if (stat.assessments.length === 0) return

      // Overall scores
      const overallScores = stat.assessments
        .map(a => a.overall_score)
      .filter(s => typeof s === 'number' && Number.isFinite(s)) as number[]
      stat.avgOverallScore = overallScores.length > 0
        ? overallScores.reduce((a, b) => a + b, 0) / overallScores.length
        : null

      // Individual dimension scores
      const scoreKeys = new Set<string>()
      stat.assessments.forEach(a => {
        if (a.scores) {
          Object.keys(a.scores).forEach(key => scoreKeys.add(key))
        }
      })
      
      scoreKeys.forEach(key => {
        const values = stat.assessments
          .map(a => a.scores?.[key])
          .filter(v => typeof v === 'number' && Number.isFinite(v)) as number[]
        if (values.length > 0) {
          stat.avgScores[key] = values.reduce((a, b) => a + b, 0) / values.length
        }
      })

      // COT analysis
      stat.assessments.forEach(a => {
        if (a.metadata?.cot_analysis) {
          stat.cotTotalCount++
          if (a.metadata.cot_analysis.cot_detected) {
            stat.cotDetectedCount++
          }
        }
      })

      // Factuality analysis
      const factualityScores = stat.assessments
        .map(a => a.metadata?.factuality_analysis?.factuality_score)
        .filter(s => typeof s === 'number' && Number.isFinite(s)) as number[]
      stat.factualityCount = factualityScores.length
      stat.avgFactualityScore = factualityScores.length > 0
        ? factualityScores.reduce((a, b) => a + b, 0) / factualityScores.length
        : 0

      // Manipulation resistance
      const resistanceScores = stat.assessments
        .map(a => a.metadata?.manipulation_resistance?.resistance_score)
        .filter(s => typeof s === 'number' && Number.isFinite(s)) as number[]
      stat.manipulationResistanceCount = resistanceScores.length
      stat.avgManipulationResistance = resistanceScores.length > 0
        ? resistanceScores.reduce((a, b) => a + b, 0) / resistanceScores.length
        : 0

      // Manipulation capability
      const capabilityScores = stat.assessments
        .map(a => a.metadata?.manipulation_capability?.capability_score)
        .filter(s => typeof s === 'number' && Number.isFinite(s)) as number[]
      stat.manipulationCapabilityCount = capabilityScores.length
      stat.avgManipulationCapability = capabilityScores.length > 0
        ? capabilityScores.reduce((a, b) => a + b, 0) / capabilityScores.length
        : 0
    })

    return Object.values(stats).sort((a, b) => {
      // Sort by selected metric or overall score
      if (selectedMetric === 'overall_score') {
        return (b.avgOverallScore ?? -Infinity) - (a.avgOverallScore ?? -Infinity)
      }
      if (selectedMetric === 'testCount') {
        return b.testCount - a.testCount
      }
      return (b.avgScores[selectedMetric] || 0) - (a.avgScores[selectedMetric] || 0)
    })
  }, [testRuns, selectedTestType, assessmentsMap, selectedMetric])

  // Chart data with analysis scores
  const chartData = modelStats.slice(0, 10).map((stat) => ({
    name: modelLabel(stat.model).length > 24 ? modelLabel(stat.model).substring(0, 24) + '…' : modelLabel(stat.model),
    fullName: modelLabel(stat.model),
    'Test Count': stat.testCount,
    'Completed': stat.completedCount,
    'Pending': stat.statusCounts.pending || 0,
    'Failed': stat.statusCounts.failed || 0,
    'Avg Overall Score': stat.avgOverallScore === null ? null : stat.avgOverallScore * 100,
    'Avg Safety': stat.avgScores.safety === undefined ? null : stat.avgScores.safety * 100,
    'Avg Alignment': stat.avgScores.alignment === undefined ? null : stat.avgScores.alignment * 100,
    'Avg Factuality': stat.factualityCount ? stat.avgFactualityScore * 100 : null,
    'Manipulation Resistance': stat.manipulationResistanceCount ? stat.avgManipulationResistance * 100 : null,
  }))

  // Analysis score comparison chart data
  const analysisScoreData = useMemo(() => {
    const scoreDimensions = ['safety', 'alignment', 'reasoning', 'jailbreak_resistance', 'ethical_reasoning', 'factuality', 'manipulation_resistance']
    return modelStats.slice(0, 8).map(stat => {
      const data: any = {
        model: modelLabel(stat.model).length > 24 ? modelLabel(stat.model).substring(0, 24) + '…' : modelLabel(stat.model),
        fullName: modelLabel(stat.model),
      }
      scoreDimensions.forEach(dim => {
        data[formatLabel(dim)] = stat.avgScores[dim] === undefined ? null : stat.avgScores[dim] * 100
      })
      return data
    })
  }, [modelStats, displayNames])

  // Radar chart data for top models - format for recharts
  const radarData = useMemo(() => {
    const topModels = modelStats.filter(stat => Object.keys(stat.avgScores).length > 0).slice(0, 5)
    const dimensions = [
      { name: 'Safety', key: 'safety' },
      { name: 'Alignment', key: 'alignment' },
      { name: 'Reasoning', key: 'reasoning' },
      { name: 'Jailbreak Resistance', key: 'jailbreak_resistance' },
      { name: 'Ethical Reasoning', key: 'ethical_reasoning' },
      { name: 'Factuality', key: 'factuality' },
    ]
    
    return topModels.map(stat => {
      const data: any = { model: modelLabel(stat.model) }
      dimensions.forEach(({ name, key }) => {
        data[name] = stat.avgScores[key] === undefined ? null : stat.avgScores[key] * 100
      })
      return data
    })
  }, [modelStats, displayNames])
  
  // Format radar data for recharts (needs array of objects with subject/value pairs)
  const radarChartData = useMemo(() => {
    if (radarData.length === 0) return []
    
    const dimensions = ['Safety', 'Alignment', 'Reasoning', 'Jailbreak Resistance', 'Ethical Reasoning', 'Factuality']
    return radarData.map(modelData => ({
      model: modelData.model as string,
      points: dimensions.map(dim => ({
        subject: dim,
        value: modelData[dim] as number | null,
      })),
    }))
  }, [radarData])

  const testTypeData = useMemo(() => {
    const typeCounts: Record<string, number> = {}
    testRuns.forEach((run) => {
      if (selectedTestType && run.test_type !== selectedTestType) return
      if (run.status !== 'completed') return
      const type = run.test_type || 'unknown'
      typeCounts[type] = (typeCounts[type] || 0) + 1
    })
    return Object.entries(typeCounts).map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }))
  }, [testRuns, selectedTestType])

  const statusData = useMemo(() => {
    const statusCounts: Record<string, number> = {}
    testRuns.forEach((run) => {
      if (selectedTestType && run.test_type !== selectedTestType) return
      statusCounts[run.status] = (statusCounts[run.status] || 0) + 1
    })
    return Object.entries(statusCounts).map(([name, value]) => ({ 
      name: name.charAt(0).toUpperCase() + name.slice(1), 
      value,
      color: name === 'completed' ? '#10b981' : name === 'running' ? '#3b82f6' : name === 'failed' ? '#ef4444' : '#6b7280'
    }))
  }, [testRuns, selectedTestType])

  const modelComparisonData = modelStats.slice(0, 5).map((stat) => ({
    model: modelLabel(stat.model).length > 24 ? modelLabel(stat.model).substring(0, 24) + '…' : modelLabel(stat.model),
    fullName: modelLabel(stat.model),
    ...Object.fromEntries(availableTestTypes.map(type => [formatLabel(type), stat.testTypes[type] || 0])),
  }))

  const columns = [
    { key: 'model', header: 'Model', render: (stat: typeof modelStats[0]) => <span title={stat.model}>{modelLabel(stat.model)}</span> },
    { key: 'provider', header: 'Provider' },
    { key: 'testCount', header: 'Total Tests' },
    { key: 'completedCount', header: 'Completed' },
    { 
      key: 'avgOverallScore', 
      header: 'Avg Overall Score',
      render: (stat: typeof modelStats[0]) => (
        <span className="font-medium">
          {stat.avgOverallScore !== null ? `${(stat.avgOverallScore * 100).toFixed(1)}%` : '-'}
        </span>
      )
    },
    { 
      key: 'avgScores', 
      header: 'Key Scores',
      render: (stat: typeof modelStats[0]) => (
        <div className="flex flex-col gap-1 text-xs">
          {stat.avgScores.safety !== undefined && (
            <span>Safety: {(stat.avgScores.safety * 100).toFixed(1)}%</span>
          )}
          {stat.avgScores.alignment !== undefined && (
            <span>Alignment: {(stat.avgScores.alignment * 100).toFixed(1)}%</span>
          )}
          {stat.avgScores.factuality !== undefined && (
            <span>Factuality: {(stat.avgScores.factuality * 100).toFixed(1)}%</span>
          )}
          {stat.factualityCount > 0 && (
            <span className="text-green-600">Factuality (AI): {(stat.avgFactualityScore * 100).toFixed(1)}%</span>
          )}
        </div>
      )
    },
    { 
      key: 'cotAnalysis', 
      header: 'COT Detection',
      render: (stat: typeof modelStats[0]) => (
        stat.cotTotalCount > 0 ? (
          <div className="text-xs">
            <span>{stat.cotDetectedCount}/{stat.cotTotalCount}</span>
            <span className="text-muted-foreground ml-1">
              ({(stat.cotDetectedCount / stat.cotTotalCount * 100).toFixed(0)}%)
            </span>
          </div>
        ) : (
          <span className="text-muted-foreground text-xs">-</span>
        )
      )
    },
    { 
      key: 'manipulation', 
      header: 'Manipulation',
      render: (stat: typeof modelStats[0]) => (
        <div className="flex flex-col gap-1 text-xs">
          {stat.manipulationResistanceCount > 0 && (
            <span>Resistance: {(stat.avgManipulationResistance * 100).toFixed(1)}%</span>
          )}
          {stat.manipulationCapabilityCount > 0 && (
            <span>Capability: {(stat.avgManipulationCapability * 100).toFixed(1)}%</span>
          )}
          {stat.manipulationResistanceCount === 0 && stat.manipulationCapabilityCount === 0 && (
            <span className="text-muted-foreground">-</span>
          )}
        </div>
      )
    },
    { 
      key: 'testTypes', 
      header: 'Test Types',
      render: (stat: typeof modelStats[0]) => (
        <div className="flex gap-1 flex-wrap">
          {Object.entries(stat.testTypes).map(([type, count]) => (
            <span key={type} className="text-xs bg-muted px-2 py-1 rounded">
              {type}: {count}
            </span>
          ))}
        </div>
      )
    },
  ]

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

  // Get all available score dimensions for filtering
  const availableDimensions = useMemo(() => {
    const dims = new Set<string>(['overall_score', 'testCount'])
    modelStats.forEach(stat => {
      Object.keys(stat.avgScores).forEach(key => dims.add(key))
    })
    return Array.from(dims).sort()
  }, [modelStats, displayNames])

  if (isLoading || loadingAssessments) {
    return (
      <div className="flex min-h-48 items-center justify-center" aria-label="Loading evaluation charts">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  return (
      <section className="space-y-6" aria-label="Evaluation charts">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold">Evaluation charts</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Behavioral evaluation scores and test coverage. Choose a test type to compare matching evaluations.
            </p>
          </div>
          <div className="flex gap-2">
            <select
              aria-label="Evaluation test type"
              value={selectedTestType}
              onChange={(e) => setSelectedTestType(e.target.value)}
              className="rounded border px-3 py-2 text-sm"
            >
              <option value="">All Test Types</option>
              {availableTestTypes.map(type => <option key={type} value={type}>{formatLabel(type)}</option>)}
            </select>
            <select
              aria-label="Evaluation sort metric"
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="rounded border px-3 py-2 text-sm"
            >
              <option value="overall_score">Sort by: Overall Score</option>
              <option value="testCount">Sort by: Test Count</option>
              {availableDimensions.filter(d => d !== 'overall_score' && d !== 'testCount').map(dim => (
                <option key={dim} value={dim}>
                  Sort by: {dim.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
        </div>

        {runsError && <div role="alert" className="rounded-lg border border-destructive/40 p-4 text-sm">
          Could not load evaluation runs: {runsError.message}. <button type="button" className="underline" onClick={() => refetchRuns()}>Retry</button>
        </div>}
        {(assessmentsError || failedAssessmentIds.length > 0) && <div role="alert" className="rounded-lg border border-amber-500/40 p-4 text-sm">
          Assessment scores could not be loaded{failedAssessmentIds.length ? ` for ${failedAssessmentIds.length} run(s)` : ''}. Charts show only scores that loaded successfully. <button type="button" className="underline" onClick={() => refetchAssessments()}>Retry scores</button>
        </div>}
        {!runsError && modelStats.length === 0 && <div className="rounded-lg border bg-muted/30 p-4 text-sm">
          No evaluation runs match this selection. Select another model, experiment, or test type, or <Link href="/create-test" className="text-primary underline">create an evaluation</Link>.
        </div>}
        {modelStats.length > 0 && !modelStats.some(stat => stat.assessments.length > 0) && !failedAssessmentIds.length && !assessmentsError && <div className="rounded-lg border bg-muted/30 p-4 text-sm">
          {testRuns.some(run => run.test_type === 'benchmark') && (!selectedTestType || selectedTestType === 'benchmark') ? <>Benchmark accuracy is available in <Link href="/benchmarks" className="text-primary underline">Benchmark scores & comparisons</Link> and requires no separate assessment. The charts below show test coverage; behavioral score charts appear when behavioral assessments are recorded.</> : <>These runs have no recorded assessment scores yet. Test coverage charts are available below. <Link href="/test-runs" className="text-primary underline">Open a completed test run</Link> to run its analysis and populate score charts.</>}
        </div>}
        <p className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
          These averages combine the recorded test runs. Compare models on the same prompts, settings, and evaluator before attributing a score difference to the weights. Missing scores are shown as unavailable, never as zero.
        </p>

        {modelStats.length > 0 && <>
        {/* Summary Cards */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-4 lg:grid-cols-6">
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">Total Models</h3>
            <p className="text-2xl font-bold">{modelStats.length}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">Total Test Runs</h3>
            <p className="text-2xl font-bold">{testRuns.filter(r => !selectedTestType || r.test_type === selectedTestType).length}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">Completed Runs</h3>
            <p className="text-2xl font-bold text-green-600">
              {testRuns.filter(r => r.status === 'completed' && (!selectedTestType || r.test_type === selectedTestType)).length}
            </p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">With Analysis</h3>
            <p className="text-2xl font-bold text-blue-600">
              {modelStats.filter(m => m.assessments.length > 0).length}
            </p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">Avg Overall Score</h3>
            <p className="text-2xl font-bold">
              {modelStats.filter(m => m.avgOverallScore !== null).length > 0
                ? `${(modelStats
                    .filter(m => m.avgOverallScore !== null)
                    .reduce((sum, m) => sum + (m.avgOverallScore ?? 0), 0) / 
                    modelStats.filter(m => m.avgOverallScore !== null).length * 100).toFixed(1)}%`
                : '-'}
            </p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">COT Detected</h3>
            <p className="text-2xl font-bold">
              {modelStats.reduce((sum, m) => sum + m.cotDetectedCount, 0)} / {modelStats.reduce((sum, m) => sum + m.cotTotalCount, 0)}
            </p>
          </div>
        </div>

        {/* Analysis Scores Section - Grouped Bar Chart for Better Comparison */}
        {modelStats.some(s => Object.keys(s.avgScores).length > 0) && (
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Analysis Score Comparison</h2>
            <p className="text-sm text-muted-foreground mb-4">
              Compare scores across different dimensions. Higher scores indicate better performance.
            </p>
            <ResponsiveContainer width="100%" height={450}>
              <BarChart data={analysisScoreData} margin={{ top: 20, right: 30, left: 20, bottom: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="model" 
                  angle={-45} 
                  textAnchor="end" 
                  height={120}
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  domain={[0, 100]} 
                  label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                  contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', border: '1px solid #ccc' }}
                />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                  iconType="square"
                />
                <Bar dataKey="Safety" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Alignment" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Reasoning" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Jailbreak Resistance" fill="#ef4444" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Ethical Reasoning" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Factuality" fill="#ec4899" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Overall Score with Dimension Breakdown */}
        {modelStats.some(s => s.avgOverallScore !== null) && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-lg border bg-card p-6">
              <h2 className="text-lg font-semibold mb-4">Overall Score Ranking</h2>
              <p className="text-sm text-muted-foreground mb-4">
                Models ranked by average overall safety score
              </p>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart 
                  data={chartData
                    .filter(d => d['Avg Overall Score'] !== null)
                    .sort((a, b) => (b['Avg Overall Score'] ?? 0) - (a['Avg Overall Score'] ?? 0))
                  }
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis 
                    type="category" 
                    dataKey="name" 
                    width={90}
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip 
                    formatter={(value: number) => `${value.toFixed(1)}%`}
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', border: '1px solid #ccc' }}
                  />
                  <Bar 
                    dataKey="Avg Overall Score" 
                    fill="#3b82f6" 
                    radius={[0, 8, 8, 0]}
                  >
                    {chartData
                      .filter(d => d['Avg Overall Score'] !== null)
                      .sort((a, b) => (b['Avg Overall Score'] ?? 0) - (a['Avg Overall Score'] ?? 0))
                      .map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={(entry['Avg Overall Score'] ?? 0) >= 80 ? '#10b981' : (entry['Avg Overall Score'] ?? 0) >= 60 ? '#3b82f6' : '#f59e0b'} 
                        />
                      ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="rounded-lg border bg-card p-6">
              <h2 className="text-lg font-semibold mb-4">Score Heatmap</h2>
              <p className="text-sm text-muted-foreground mb-4">
                Visual comparison across all dimensions
              </p>
              <div className="space-y-2 max-h-[350px] overflow-y-auto">
                {analysisScoreData.slice(0, 8).map((model, idx) => {
                  const dimensions = ['Safety', 'Alignment', 'Reasoning', 'Jailbreak Resistance', 'Ethical Reasoning', 'Factuality']
                  const maxScore = Math.max(...dimensions.map(d => model[d] || 0))
                  
                  return (
                    <div key={idx} className="border-b pb-3 last:border-b-0">
                      <div className="text-sm font-medium mb-2">{model.fullName || model.model}</div>
                      <div className="space-y-1">
                        {dimensions.map((dim, dimIdx) => {
                          const score = model[dim] as number | null
                          const percentage = maxScore > 0 && score !== null ? (score / maxScore) * 100 : 0
                          const colorIntensity = Math.min(percentage / 100, 1)
                          const colors = [
                            { r: 59, g: 130, b: 246 },   // Safety - blue
                            { r: 16, g: 185, b: 129 },   // Alignment - green
                            { r: 245, g: 158, b: 11 },   // Reasoning - amber
                            { r: 239, g: 68, b: 68 },    // Jailbreak Resistance - red
                            { r: 139, g: 92, b: 246 },   // Ethical Reasoning - purple
                            { r: 236, g: 72, b: 153 },   // Factuality - pink
                          ]
                          const color = colors[dimIdx]
                          const bgColor = `rgba(${color.r}, ${color.g}, ${color.b}, ${0.2 + colorIntensity * 0.6})`
                          
                          return (
                            <div key={dim} className="flex items-center gap-2">
                              <div className="text-xs text-muted-foreground w-32 flex-shrink-0">
                                {dim}
                              </div>
                              <div className="flex-1 h-5 bg-muted rounded-full overflow-hidden relative">
                                <div 
                                  className="h-full rounded-full transition-all"
                                  style={{ 
                                    width: `${score ?? 0}%`,
                                    backgroundColor: bgColor,
                                  }}
                                />
                                <span className="absolute inset-0 flex items-center justify-center text-xs font-medium text-foreground">
                                  {score !== null ? `${score.toFixed(0)}%` : 'N/A'}
                                </span>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}

        {/* Radar Chart for Top Models */}
        {radarChartData.length > 0 && (
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Top Models - Multi-Dimensional Analysis</h2>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
              {radarChartData.slice(0, 6).map((modelData, idx) => (
                <div key={modelData.model} className="flex flex-col items-center">
                  <h3 className="text-sm font-medium mb-2">{modelData.model}</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <RadarChart data={modelData.points}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="subject" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar 
                        name={modelData.model} 
                        dataKey="value" 
                        stroke={COLORS[idx % COLORS.length]} 
                        fill={COLORS[idx % COLORS.length]} 
                        fillOpacity={0.6} 
                      />
                      <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Charts Row 1 */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Model Test Count</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Test Count" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Test Run Status Distribution</h2>
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
        </div>

        {/* Charts Row 2 */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Test Type Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={testTypeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Model Status Breakdown</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Completed" stackId="a" fill="#10b981" />
                <Bar dataKey="Pending" stackId="a" fill="#f59e0b" />
                <Bar dataKey="Failed" stackId="a" fill="#ef4444" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Factuality and Manipulation Analysis */}
        {(modelStats.some(s => s.factualityCount > 0) || modelStats.some(s => s.manipulationResistanceCount > 0)) && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {modelStats.some(s => s.factualityCount > 0) && (
              <div className="rounded-lg border bg-card p-6">
                <h2 className="text-lg font-semibold mb-4">Factuality Scores</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData.filter(d => d['Avg Factuality'] !== null)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                    <Bar dataKey="Avg Factuality" fill="#ec4899" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {modelStats.some(s => s.manipulationResistanceCount > 0) && (
              <div className="rounded-lg border bg-card p-6">
                <h2 className="text-lg font-semibold mb-4">Manipulation Resistance</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData.filter(d => d['Manipulation Resistance'] !== null)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                    <Bar dataKey="Manipulation Resistance" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* Charts Row 3 */}
        {modelComparisonData.length > 0 && (
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Top Models by Test Type</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                {availableTestTypes.filter(type => modelStats.some(stat => stat.testTypes[type])).map((type, index) => (
                  <Bar key={type} dataKey={formatLabel(type)} stackId="a" fill={COLORS[index % COLORS.length]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Model Statistics Table */}
        <div className="rounded-lg border bg-card p-6">
          <h2 className="text-lg font-semibold mb-4">Model Statistics</h2>
          <DataTable
            data={modelStats}
            columns={columns}
            emptyMessage="No model data available"
          />
        </div>
        </>}
      </section>
  )
}
