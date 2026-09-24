import { useRouter } from 'next/router'
import { ArrowLeft, ExternalLink, Square } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { MetricCard } from '@/components/common/MetricCard'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import * as Tabs from '@radix-ui/react-tabs'

import { CircuitPanel } from '@/components/interp/CircuitPanel'
import {
  useInterpArtifacts,
  useInterpRun,
  useInterpRunProgress,
  useStopInterpRun,
} from '@/lib/hooks'
import { apiClient } from '@/lib/api'
import { toast } from '@/lib/toast'

export default function InterpRunDetailPage() {
  const router = useRouter()
  const { id } = router.query
  const runId = typeof id === 'string' ? parseInt(id) : 0

  const { data: run, isLoading, error } = useInterpRun(runId)
  const stopRun = useStopInterpRun()

  const isActive = run?.status === 'running' || run?.status === 'pending'
  const { progress, isConnected } = useInterpRunProgress(runId, isActive && runId > 0)
  // Which JSON payloads this run actually produced. Most are conditional on a
  // capture flag, so the Circuit tab is offered only when there is data behind it.
  const { data: artifacts } = useInterpArtifacts(runId, run?.status === 'completed')

  if (isLoading) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </Layout>
    )
  }

  if (error || !run) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <h2 className="mb-2 text-xl font-semibold">Error loading analysis</h2>
            <p className="text-muted-foreground">
              {error instanceof Error ? error.message : 'Analysis not found'}
            </p>
          </div>
        </div>
      </Layout>
    )
  }

  const summary = run.metadata?.summary ?? {}
  const dashboardUrl = apiClient.interpDashboardUrl(run.id)
  const hasCircuitData = (artifacts?.artifacts ?? []).some((a) =>
    ['circuit_payload.json', 'minimal_circuit_payload.json'].includes(a.name),
  )

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => router.push('/interp')}
              className="text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            <div>
              <h1 className="text-3xl font-bold">
                Analysis #{run.id}
                <span className="ml-3 text-base font-normal text-muted-foreground">
                  {run.mode}
                </span>
              </h1>
              <p className="font-mono text-xs text-muted-foreground">
                {run.model_a}
                {run.model_b ? `  vs  ${run.model_b}` : ''}
              </p>
            </div>
            <StatusBadge status={run.status} />
          </div>

          <div className="flex gap-2">
            {isActive && (
              <button
                type="button"
                onClick={async () => {
                  try {
                    await stopRun.mutateAsync(run.id)
                    toast.success('Stop requested')
                  } catch (e: any) {
                    toast.error(e?.message || 'Could not stop the run')
                  }
                }}
                className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-muted"
              >
                <Square className="h-4 w-4" />
                Stop
              </button>
            )}
            {run.status === 'completed' && (
              <a
                href={dashboardUrl}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-muted"
              >
                <ExternalLink className="h-4 w-4" />
                Open dashboard
              </a>
            )}
          </div>
        </div>

        {isActive && (
          <div className="rounded-lg border bg-blue-50 p-4 dark:bg-blue-950">
            <div className="mb-2 flex items-center gap-2">
              <div
                className={`h-2 w-2 rounded-full ${
                  isConnected ? 'animate-pulse bg-green-500' : 'bg-gray-400'
                }`}
              />
              <span className="text-sm font-medium">
                {isConnected ? 'Live monitoring active' : 'Connecting…'}
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              {progress?.message ?? 'Waiting for the analysis to start…'}
            </p>
          </div>
        )}

        {run.status === 'failed' && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="mb-1 text-sm font-medium">Analysis failed</p>
            <p className="font-mono text-sm text-muted-foreground">{run.error}</p>
          </div>
        )}

        {run.status === 'completed' && (
          <>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <MetricCard
                title="Largest divergence"
                value={summary.spike_layer ?? '—'}
                subtitle="layer index"
              />
              <MetricCard title="Layers analyzed" value={summary.num_layers ?? '—'} />
              <MetricCard
                title="Window"
                value={summary.window_len ?? '—'}
                subtitle="tokens compared"
              />
              <MetricCard
                title="Shared unembedding"
                value={
                  summary.shared_unembedding === undefined || summary.shared_unembedding === null
                    ? 'n/a'
                    : summary.shared_unembedding
                      ? 'yes'
                      : 'no'
                }
                subtitle={
                  summary.shared_unembedding === false
                    ? 'logit values not comparable'
                    : undefined
                }
              />
            </div>

            {summary.shared_unembedding === false && (
              <div className="rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 text-sm dark:bg-yellow-950">
                These two models decode through different unembeddings, which happens when a
                model with tied embeddings is edited. Read the overlap of the top-k predictions
                rather than comparing probabilities directly.
              </div>
            )}

            <Tabs.Root defaultValue="dashboard">
              <Tabs.List className="flex gap-1 border-b">
                <Tabs.Trigger
                  value="dashboard"
                  className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
                >
                  Dashboard
                </Tabs.Trigger>
                {hasCircuitData && (
                  <Tabs.Trigger
                    value="circuit"
                    className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
                  >
                    Circuit
                  </Tabs.Trigger>
                )}
              </Tabs.List>

              <Tabs.Content value="dashboard" className="pt-4">
                <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
                  <div className="flex items-center justify-between border-b p-3">
                    <h2 className="text-sm font-semibold">Dashboard</h2>
                    <span className="text-xs text-muted-foreground">
                      plotly.js is served locally, so this renders offline
                    </span>
                  </div>
                  <iframe
                    src={dashboardUrl}
                    title={`Interpretability dashboard for run ${run.id}`}
                    className="h-[900px] w-full border-0 bg-white"
                  />
                </div>
              </Tabs.Content>

              {hasCircuitData && (
                <Tabs.Content value="circuit" className="pt-4">
                  <CircuitPanel runId={run.id} />
                </Tabs.Content>
              )}
            </Tabs.Root>
          </>
        )}
      </div>
    </Layout>
  )
}
