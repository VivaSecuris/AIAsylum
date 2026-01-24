import { useRouter } from 'next/router'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { SafetyScoreCard } from '@/components/analysis/SafetyScoreCard'
import { useAssessments, useRunAnalysis, useTestRun } from '@/lib/hooks'
import { formatDate } from '@/lib/utils'
import { Play } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { toast } from '@/lib/toast'

export default function AnalysisPage() {
  const router = useRouter()
  const { testRunId } = router.query
  const id = typeof testRunId === 'string' ? parseInt(testRunId) : 0

  const { data: testRun, isLoading: loadingRun } = useTestRun(id)
  const { data: assessments = [], isLoading } = useAssessments(id)
  const runAnalysis = useRunAnalysis()

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

  const handleRunAnalysis = async () => {
    if (!id) return
    try {
      await runAnalysis.mutateAsync({
        testRunId: id,
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

  const chartData = assessments[0]?.scores
    ? Object.entries(assessments[0].scores).map(([key, value]) => ({
        name: key.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
        value: value * 100,
      }))
    : []

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Analysis Dashboard</h1>
            {testRun && (
              <p className="text-sm text-muted-foreground mt-1">
                Test Run #{testRun.id} • {testRun.test_type}
              </p>
            )}
          </div>
          <button
            onClick={handleRunAnalysis}
            disabled={runAnalysis.isPending}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Play className="h-4 w-4" />
            {runAnalysis.isPending ? 'Running...' : 'Run Analysis'}
          </button>
        </div>

        {assessments.length > 0 ? (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {assessments.map((assessment) => (
                <SafetyScoreCard key={assessment.id} assessment={assessment} />
              ))}
            </div>

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
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-lg border bg-card p-12 text-center">
            <p className="text-muted-foreground mb-4">No analysis available yet.</p>
            <button
              onClick={handleRunAnalysis}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              Run Analysis
            </button>
          </div>
        )}
      </div>
    </Layout>
  )
}
