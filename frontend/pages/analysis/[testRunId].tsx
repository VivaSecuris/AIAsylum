import { useRouter } from 'next/router'
import { useState } from 'react'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { SafetyScoreCard } from '@/components/analysis/SafetyScoreCard'
import { AnalysisConfigDialog, AnalysisConfig } from '@/components/analysis/AnalysisConfigDialog'
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
  const [showAnalysisDialog, setShowAnalysisDialog] = useState(false)

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
    if (!id) return
    try {
      await runAnalysis.mutateAsync({
        testRunId: id,
        config: {
          evaluator_provider: config.evaluator_provider,
          evaluator_model: config.evaluator_model,
          enable_cot_detection: config.enable_cot_detection,
          cot_analysis_mode: config.cot_analysis_mode,
          enable_factuality_check: config.enable_factuality_check,
          enable_manipulation_analysis: config.enable_manipulation_analysis,
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
            onClick={() => setShowAnalysisDialog(true)}
            disabled={runAnalysis.isPending}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Play className="h-4 w-4" />
            {runAnalysis.isPending ? 'Running...' : 'Run Analysis'}
          </button>
        </div>

        {assessments.length > 0 ? (
          <div className="space-y-6 overflow-y-auto max-h-[calc(100vh-250px)]">
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

      <AnalysisConfigDialog
        isOpen={showAnalysisDialog}
        onClose={() => setShowAnalysisDialog(false)}
        onConfirm={handleRunAnalysis}
        doctorProvider={testRun?.doctor_provider}
        doctorModel={testRun?.doctor_model}
      />
    </Layout>
  )
}
