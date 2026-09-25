import { useRouter } from 'next/router'
import { ArrowLeft, ExternalLink, Square } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { MetricCard } from '@/components/common/MetricCard'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import * as Tabs from '@radix-ui/react-tabs'

import { CircuitPanel } from '@/components/interp/CircuitPanel'
import { PatchingPanel } from '@/components/interp/PatchingPanel'
import { RunContext } from '@/components/interp/RunContext'
import {
  useInterpArtifacts,
  useInterpRun,
  useInterpRunProgress,
  useStopInterpRun,
} from '@/lib/hooks'
import { apiClient } from '@/lib/api'
import { toast } from '@/lib/toast'
import { formatApiError } from '@/lib/utils'

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
  const { data: artifacts, error: artifactsError } = useInterpArtifacts(runId, run?.status === 'completed')

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
              {error ? formatApiError(error) : 'Analysis not found'}
            </p>
          </div>
        </div>
      </Layout>
    )
  }

  const summary = run.metadata?.summary ?? {}
  const dashboardUrl = apiClient.interpDashboardUrl(run.id)
  const artifactNames = artifacts?.artifacts.map((artifact) => artifact.name)
  const hasPatching = artifactNames?.includes('patching_results.json')
  const isComparison = run.mode === 'comparison' || run.mode === 'model_diff'
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
              aria-label="Back to analyses"
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
                    toast.error(formatApiError(e, 'Could not stop the run'))
                  }
                }}
                disabled={stopRun.isPending}
                className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-muted"
              >
                <Square className="h-4 w-4" />
                {stopRun.isPending ? 'Stopping…' : 'Stop'}
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
                {isConnected ? 'Live monitoring active' : 'Live stream reconnecting; status refreshes every 5 seconds'}
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              {progress?.message ?? (run.status === 'pending' ? 'Queued for the server’s shared model slot. Other model analyses must finish first.' : 'The model is loading or the analysis is running. The first run may download weights.')}
            </p>
          </div>
        )}

        {run.status === 'failed' && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="mb-1 text-sm font-medium">{run.metadata?.cancelled ? 'Analysis stopped' : 'Analysis failed'}</p>
            <p className="font-mono text-sm text-muted-foreground">{run.error}</p>
            <p className="mt-2 text-xs text-muted-foreground">For memory errors, reduce the sequence limit or captures, or use a smaller model or a device with more available memory. Check model fit again before retrying.</p>
          </div>
        )}

        <RunContext run={run} artifactNames={artifactNames} />

        {run.status === 'completed' && (
          <>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              {isComparison && <MetricCard
                title="Largest divergence"
                value={summary.spike_layer ?? '—'}
                subtitle="hidden-state index"
              />}
              <MetricCard title="Hidden-state snapshots" value={summary.num_layers ?? '—'} subtitle="includes the embedding state" />
              <MetricCard
                title="Window"
                value={summary.window_len ?? '—'}
                subtitle="tokens compared"
              />
              {run.mode === 'model_diff' && <MetricCard
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
              />}
            </div>

            {summary.shared_unembedding === false && (
              <div className="rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 text-sm dark:bg-yellow-950">
                These two models decode through different unembeddings, which happens when a
                model with tied embeddings is edited. Read the overlap of the top-k predictions
                rather than comparing probabilities directly.
              </div>
            )}

            {artifactsError && <p role="alert" className="rounded border p-3 text-sm text-destructive">{formatApiError(artifactsError, 'Could not load the artifact list')}. The dashboard is still available below.</p>}

            <Tabs.Root defaultValue="dashboard">
              <Tabs.List className="flex flex-wrap gap-1 border-b" aria-label="Analysis results">
                <Tabs.Trigger
                  value="dashboard"
                  className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
                >
                  Dashboard
                </Tabs.Trigger>
                {hasPatching && <Tabs.Trigger value="patching" className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary">Activation patching</Tabs.Trigger>}
                {hasCircuitData && (
                  <Tabs.Trigger
                    value="circuit"
                    className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
                  >
                    Exploratory circuits
                  </Tabs.Trigger>
                )}
              </Tabs.List>

              <Tabs.Content value="dashboard" className="pt-4">
                <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
                  <div className="flex items-center justify-between border-b p-3">
                    <h2 className="text-sm font-semibold">Dashboard</h2>
                    <span className="text-xs text-muted-foreground">
                      Use the charts to inspect layers and aligned token positions
                    </span>
                  </div>
                  <iframe
                    src={dashboardUrl}
                    title={`Interpretability dashboard for run ${run.id}`}
                    className="h-[900px] w-full border-0 bg-white"
                  />
                </div>
              </Tabs.Content>

              {hasPatching && <Tabs.Content value="patching" className="pt-4"><PatchingPanel runId={run.id} /></Tabs.Content>}

              {hasCircuitData && (
                <Tabs.Content value="circuit" className="pt-4">
                  <CircuitPanel runId={run.id} artifactNames={artifactNames ?? []} />
                </Tabs.Content>
              )}
            </Tabs.Root>
          </>
        )}
      </div>
    </Layout>
  )
}
