import { useRouter } from 'next/router'
import { ArrowLeft, Square } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { MetricCard } from '@/components/common/MetricCard'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { CompareInInterpButton } from '@/components/weights/CompareInInterpButton'
import { CompareTable } from '@/components/weights/CompareTable'
import { FrontierChart } from '@/components/weights/FrontierChart'
import { LayerScoreChart } from '@/components/weights/LayerScoreChart'
import { ManifestCard, ProvenanceCard } from '@/components/weights/ManifestCard'
import { SweepTable } from '@/components/weights/SweepTable'
import {
  useEditedModel,
  useStopWeightRun,
  useWeightRun,
  useWeightRunProgress,
} from '@/lib/hooks'
import { toast } from '@/lib/toast'

export default function WeightRunDetailPage() {
  const router = useRouter()
  const { id } = router.query
  const runId = typeof id === 'string' ? parseInt(id) : 0

  // Declared before any early return: hooks must not be conditional.
  const { data: run, isLoading, error } = useWeightRun(runId, true)
  const stopRun = useStopWeightRun()
  const isActive = run?.status === 'running' || run?.status === 'pending'
  const { progress, isConnected } = useWeightRunProgress(runId, isActive && runId > 0)

  const summary = run?.metadata?.summary ?? {}
  const outName = run?.out_dir ? run.out_dir.split('/').filter(Boolean).pop() ?? '' : ''
  const { data: modelDetail } = useEditedModel(
    run?.kind === 'surgery' && run?.status === 'completed' ? outName : '',
  )

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
            <h2 className="mb-2 text-xl font-semibold">Error loading run</h2>
            <p className="text-muted-foreground">
              {error instanceof Error ? error.message : 'Run not found'}
            </p>
          </div>
        </div>
      </Layout>
    )
  }

  const sourceDirection = run.metadata?.source_direction ?? {}

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => router.push('/weights')}
              className="text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            <div>
              <h1 className="text-3xl font-bold">
                {{
                  direction: 'Direction',
                  sweep: 'Causal check',
                  select: 'Capability search',
                  surgery: 'Weight surgery',
                  compare: 'Measurement',
                }[run.kind] ?? run.kind}{' '}
                #{run.id}
              </h1>
              <p className="font-mono text-xs text-muted-foreground">
                {run.source_model}
                {run.method ? ` · ${run.method}` : ''}
                {run.objective ? ` · ${run.objective}` : ''}
              </p>
            </div>
            <StatusBadge status={run.status} />
          </div>

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
              {progress?.message ?? 'Waiting to start…'}
            </p>
            {progress?.event_type === 'weights_queued' && (
              <p className="mt-1 text-xs text-muted-foreground">
                Only one model-heavy job runs at a time, so this is queued rather than stuck.
              </p>
            )}
          </div>
        )}

        {run.status === 'failed' && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="mb-1 text-sm font-medium">Run failed</p>
            <p className="font-mono text-sm text-muted-foreground">{run.error}</p>
          </div>
        )}

        {/* The direction this stage consumed, snapshotted at creation so it
            still renders after the parent run is deleted. */}
        {run.kind !== 'direction' && sourceDirection.layer != null && (
          <div className="rounded-lg border bg-card p-4 text-sm shadow-sm">
            <p className="text-xs text-muted-foreground">Using direction</p>
            <p className="font-mono text-xs">
              run #{run.source_run_id} · layer {sourceDirection.layer} · AUC{' '}
              {Number(sourceDirection.auc).toFixed(3)} · from {sourceDirection.model_id}
            </p>
          </div>
        )}

        {run.status === 'completed' && run.kind === 'direction' && (
          <>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <MetricCard title="Best layer" value={summary.layer ?? '—'} subtitle="chosen on held-out" />
              <MetricCard
                title="Held-out AUC"
                value={summary.auc != null ? Number(summary.auc).toFixed(4) : '—'}
                subtitle={`threshold ${summary.min_usable_auc ?? 0.9}`}
              />
              <MetricCard
                title="Cohen's d"
                value={summary.cohens_d != null ? Number(summary.cohens_d).toFixed(2) : '—'}
                subtitle="tiebreak when AUC saturates"
              />
              <MetricCard title="d_model" value={summary.d_model ?? '—'} subtitle={summary.split_hash} />
            </div>

            <div
              className={`rounded-lg border p-4 text-sm ${
                summary.usable
                  ? 'border-emerald-500/50 bg-emerald-50 dark:bg-emerald-950'
                  : 'border-destructive/50 bg-destructive/10'
              }`}
            >
              {summary.usable ? (
                <>
                  <p className="font-medium">Usable direction</p>
                  <p className="text-muted-foreground">
                    It separates the classes on data it was not fitted to. Next, prove it is
                    causal with a steering sweep — a direction that separates is not
                    necessarily one the model uses.
                  </p>
                </>
              ) : (
                <>
                  <p className="font-medium">Below the usability threshold</p>
                  <p className="text-muted-foreground">
                    This direction does not separate the classes, so steering and surgery will
                    not produce a reliable behaviour change. Deriving again with a cleaner
                    contrast is the fix, not pushing on.
                  </p>
                </>
              )}
            </div>

            <LayerScoreChart
              layerScores={summary.layer_scores ?? []}
              selectedLayer={summary.layer}
              minUsableAuc={summary.min_usable_auc ?? 0.9}
            />

            {summary.split && (
              <div className="rounded-lg border bg-card p-6 shadow-sm">
                <h2 className="mb-2 text-lg font-semibold">Prompt split</h2>
                <p className="mb-3 text-sm text-muted-foreground">
                  Seeded and hashed, so the same split can be rebuilt exactly. The direction was
                  fitted on train and the layer chosen on test.
                </p>
                <pre className="overflow-auto rounded bg-muted/50 p-3 font-mono text-xs">
                  {JSON.stringify(summary.split, null, 2)}
                </pre>
              </div>
            )}
          </>
        )}

        {run.status === 'completed' && run.kind === 'sweep' && (
          <SweepTable
            rows={summary.rows ?? []}
            verdict={summary.verdict}
            ablateDeltaPoints={summary.ablate_delta_points}
            factualDeltaPoints={summary.factual_delta_points}
            layer={summary.layer}
          />
        )}

        {run.status === 'completed' && run.kind === 'select' && (
          <>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <MetricCard
                title="Recommended rank"
                value={summary.best?.rank ?? '—'}
                subtitle={summary.best ? `k = ${Number(summary.best.k).toFixed(2)}` : 'none admissible'}
              />
              <MetricCard
                title="Candidates scored"
                value={summary.frontier?.length ?? '—'}
              />
              <MetricCard
                title="Held the floor"
                value={
                  summary.frontier
                    ? summary.frontier.filter((r: any) => r.accepted).length
                    : '—'
                }
                subtitle={`floor ${((summary.factual_floor ?? 0) * 100).toFixed(0)} pts`}
              />
              <MetricCard
                title="Subspace rank"
                value={summary.source_rank ?? '—'}
                subtitle="available to search"
              />
            </div>

            <FrontierChart
              baseline={summary.baseline}
              frontier={summary.frontier ?? []}
              best={summary.best}
              factualFloor={summary.factual_floor ?? 0.05}
            />

            <p className="text-sm text-muted-foreground">
              Nothing here was written to disk. To keep a configuration, derive the subspace at
              that rank so the basis matches, then run the surgery stage with that strength.
            </p>
          </>
        )}

        {run.status === 'completed' && run.kind === 'compare' && summary.metrics && (
          <CompareTable
            metrics={summary.metrics}
            deltas={summary.deltas}
            responses={summary.responses}
            prompts={summary.prompts}
          />
        )}

        {run.status === 'completed' && run.kind === 'surgery' && (
          <>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <MetricCard title="Beta" value={summary.manifest?.beta ?? '—'} subtitle="0 removes, 1 no-op, 2 amplifies" />
              <MetricCard title="Matrices edited" value={summary.manifest?.matrices_edited ?? '—'} />
              <MetricCard
                title="Mean relative change"
                value={
                  summary.manifest?.mean_relative_change != null
                    ? Number(summary.manifest.mean_relative_change).toFixed(4)
                    : '—'
                }
              />
              <MetricCard
                title="On disk"
                value={
                  summary.size_bytes ? `${(summary.size_bytes / 1024 ** 3).toFixed(2)} GB` : '—'
                }
              />
            </div>

            <CompareInInterpButton
              sourceModel={run.source_model}
              editedPath={run.out_dir ?? ''}
              probePrompts={sourceDirection.probe_prompts ?? []}
            />

            {summary.manifest && <ManifestCard manifest={summary.manifest} />}
            {modelDetail && <ProvenanceCard detail={modelDetail} />}

            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <h2 className="mb-2 text-lg font-semibold">Use it in a test run</h2>
              <p className="mb-3 text-sm text-muted-foreground">
                Serve it through the transformers provider. Run the baseline through the same
                provider too — comparing this against a stock model served through Ollama would
                measure the serving stack as much as the weights.
              </p>
              <pre className="overflow-auto rounded bg-muted/50 p-3 font-mono text-xs">
{`provider: transformers
model:    ${run.out_dir}`}
              </pre>
            </div>
          </>
        )}
      </div>
    </Layout>
  )
}
