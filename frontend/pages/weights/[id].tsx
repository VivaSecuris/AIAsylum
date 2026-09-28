import { useRouter } from 'next/router'
import Link from 'next/link'
import { ArrowLeft, Square } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { MetricCard } from '@/components/common/MetricCard'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { CompareInInterpButton } from '@/components/weights/CompareInInterpButton'
import { CompareTable } from '@/components/weights/CompareTable'
import { FrontierChart } from '@/components/weights/FrontierChart'
import { TrialLog } from '@/components/weights/TrialLog'
import { VerificationCard } from '@/components/weights/VerificationCard'
import { LayerScoreChart } from '@/components/weights/LayerScoreChart'
import { ManifestCard, ProvenanceCard } from '@/components/weights/ManifestCard'
import { SweepTable } from '@/components/weights/SweepTable'
import { SubspaceCurveChart } from '@/components/weights/SubspaceCurveChart'
import { RefusalTimelineChart } from '@/components/weights/RefusalTimelineChart'
import { ProbeResults } from '@/components/weights/ProbeResults'
import { RoutingHeatmap } from '@/components/weights/RoutingHeatmap'
import { TrainingCurve } from '@/components/weights/TrainingCurve'
import { ResultSummary } from '@/components/weights/ResultSummary'
import { LiveStepView } from '@/components/weights/LiveStepView'
import { InduceSection } from '@/components/weights/InduceSection'
import { HNeuronResults } from '@/components/weights/HNeuronResults'
import { RedteamLeakBars } from '@/components/weights/RedteamLeakBars'
import { EmbedResults } from '@/components/weights/EmbedResults'
import { WRITING_WEIGHT_KINDS } from '@/lib/api'
import {
  useEditedModel,
  useRoutingStats,
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
  const { progress, history, isConnected } = useWeightRunProgress(runId, isActive && runId > 0)

  const summary = run?.metadata?.summary ?? {}
  const outName = run?.out_dir ? run.out_dir.split('/').filter(Boolean).pop() ?? '' : ''
  const writesModel = !!run && WRITING_WEIGHT_KINDS.includes(run.kind)
  const training = run?.kind === 'lora' || run?.kind === 'distill'
  // A training run that kept only its adapter has no model directory to look up.
  const producedModel = writesModel && (!training || summary.merged === true)
  const { data: modelDetail } = useEditedModel(
    producedModel && run?.status === 'completed' ? outName : '',
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
                  autotune: 'Verified edit',
                  surgery: 'Weight surgery',
                  compare: 'Measurement',
                  probe: 'Harmful-intent probe',
                  routing: 'Expert routing',
                  expert_surgery: 'Expert surgery',
                  lora: 'LoRA fine-tune',
                  distill: 'Distillation',
                  induce: 'Add a targeted refusal',
                  hneurons: 'Hallucination neurons',
                  hneuron_bake: 'H-neuron bake',
                  redteam: 'Red-team a control',
                  embed_align: 'Align embeddings',
                  embed_extract: 'Reverse-engineer embeddings',
                  embed_recon: 'Reconstruct embeddings',
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
          <>
            <LiveStepView history={history} progress={progress} isConnected={isConnected} />
            {progress?.event_type === 'weights_queued' && (
              <p className="text-xs text-muted-foreground">
                Only one model-heavy job runs at a time, so this is queued rather than stuck.
              </p>
            )}
          </>
        )}

        {run.status === 'failed' && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="mb-1 text-sm font-medium">Run failed</p>
            <p className="font-mono text-sm text-muted-foreground">{run.error}</p>
          </div>
        )}

        {/* Plain-language verdict before the charts, for every completed run. */}
        {run.status === 'completed' && <ResultSummary kind={run.kind} summary={summary} />}

        {run.status === 'completed' && run.kind === 'induce' && <InduceSection summary={summary} />}
        {run.status === 'completed' && run.kind === 'hneurons' && <HNeuronResults summary={summary} />}
        {run.status === 'completed' && run.kind === 'redteam' && <RedteamLeakBars summary={summary} />}
        {run.status === 'completed' && (run.kind === 'embed_align' || run.kind === 'embed_extract' || run.kind === 'embed_recon') && (
          <EmbedResults kind={run.kind} summary={summary} />
        )}

        {summary.evaluation && <div className="rounded-lg border bg-card p-4 text-sm">
          <p>Evaluated {summary.evaluation.actual} saved held-out prompts (requested {summary.evaluation.requested}) · split {summary.evaluation.split_hash}</p>
          {(summary.evaluation.warnings ?? []).map((warning: string) => <p key={warning} className="mt-1 text-amber-700 dark:text-amber-300">{warning}</p>)}
        </div>}

        {run.status === 'completed' && run.kind === 'probe' && <ProbeResults summary={summary} />}

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
              <MetricCard
                title="Stable rank"
                value={
                  summary.extra?.stable_rank?.at_layer != null
                    ? Number(summary.extra.stable_rank.at_layer).toFixed(1)
                    : '—'
                }
                subtitle={
                  summary.extra?.stable_rank?.band
                    ? `${summary.extra.stable_rank.band} — ${summary.extra.stable_rank.note}`
                    : 'of refusal residuals at the chosen layer'
                }
              />
              <MetricCard
                title="Method"
                value={summary.method === 'rfm_agop' ? 'RFM-AGOP cone' : 'diff-in-means'}
                subtitle={
                  summary.rank && summary.rank > 1
                    ? `rank ${summary.rank}${summary.weights ? ` · weights ${summary.weights.map((w: number) => w.toFixed(2)).join(', ')}` : ''}`
                    : 'single direction'
                }
              />
              {summary.refusal_overlap && (
                <MetricCard
                  title="Overlap with refusal"
                  value={Number(summary.refusal_overlap.cosine_with_refusal).toFixed(2)}
                  subtitle={summary.refusal_overlap.interpretation}
                />
              )}
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

        {run.status === 'completed' && run.kind === 'sweep' && summary.curve && (
          <SubspaceCurveChart rows={summary.curve} summary={summary} />
        )}
        {run.status === 'completed' && run.kind === 'sweep' && summary.timelines?.length > 0 && (
          <RefusalTimelineChart timelines={summary.timelines} />
        )}
        {run.status === 'completed' && run.kind === 'sweep' && (summary.rows?.length ?? 0) > 0 && (
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
              Nothing here was written to disk. To keep a configuration, run the surgery stage
              with that rank and strength (the basis is cut to its first rows), or{' '}
              <Link
                href={{ pathname: '/weights', query: { kind: 'autotune', source_model: run.source_model, source_run_id: run.source_run_id ?? '' } }}
                className="underline"
              >
                let the autotune stage
              </Link>{' '}
              write the best admissible edit and verify it from disk.
            </p>
          </>
        )}

        {run.kind === 'autotune' && summary.trials && (
          <AutotuneSection summary={summary} />
        )}

        {run.status === 'completed' && run.kind === 'compare' && summary.metrics && (
          <CompareTable
            metrics={summary.metrics}
            deltas={summary.deltas}
            responses={summary.responses}
            prompts={summary.prompts}
            factualFloor={summary.factual_floor}
            languageDriftMax={summary.language_drift_max}
            generation={summary.generation}
            verdict={summary.verdict}
          />
        )}

        {run.status === 'completed' && run.kind === 'routing' && (
          <RoutingSection runId={run.id} sourceModel={run.source_model} summary={summary} />
        )}

        {run.status === 'completed' && training && (
          <TrainingSection run={run} summary={summary} />
        )}

        {run.status === 'completed' && writesModel && (
          <>
            {summary.manifest?.coverage_verified === false && (
              <div className="rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 text-sm dark:bg-yellow-950">
                <p className="font-medium">
                  Partial edit: {summary.manifest?.extra?.experts_edited ?? '?'} expert
                  {summary.manifest?.extra?.experts_edited === 1 ? '' : 's'} in{' '}
                  {summary.manifest?.extra?.layers_edited ?? '?'} layer
                  {summary.manifest?.extra?.layers_edited === 1 ? '' : 's'}
                </p>
                <p className="mt-1 text-muted-foreground">
                  Only the experts you named were touched. Every other expert still writes what it
                  carries when the router picks it, so read this as an expert-level intervention rather
                  than a removal of the direction.
                </p>
              </div>
            )}
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              {run.kind === 'expert_surgery' ? (
                <MetricCard
                  title={summary.manifest?.extra?.expert_mode === 'ablate' ? 'Expert scale' : 'Expert edit'}
                  value={
                    summary.manifest?.extra?.expert_mode === 'ablate'
                      ? summary.manifest?.extra?.expert_scale ?? '—'
                      : summary.manifest?.beta ?? (summary.manifest?.extra?.k != null ? `k=${summary.manifest.extra.k}` : '—')
                  }
                  subtitle={summary.manifest?.extra?.expert_mode === 'ablate' ? '0 removes their write, 1 no-op' : summary.manifest?.extra?.expert_mode}
                />
              ) : run.kind === 'surgery' ? (
                <MetricCard title="Beta" value={summary.manifest?.beta ?? '—'} subtitle="0 removes, 1 no-op, 2 amplifies" />
              ) : run.kind === 'autotune' ? (
                <MetricCard
                  title="Edit written"
                  value={summary.winner ? `rank ${summary.winner.rank}, k ${Number(summary.winner.k).toFixed(2)}` : '—'}
                  subtitle={summary.winner ? `embeddings ${summary.winner.include_embeddings ? 'edited' : 'untouched'}` : undefined}
                />
              ) : (
                <MetricCard title="Method" value={summary.manifest?.method ?? run.method ?? '—'} subtitle={summary.manifest?.extra?.distill?.teacher ? `from ${summary.manifest.extra.distill.teacher}` : undefined} />
              )}
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

            {producedModel ? (
              <>
                <CompareInInterpButton
                  sourceModel={run.source_model}
                  editedPath={run.out_dir ?? ''}
                  probePrompts={sourceDirection.probe_prompts ?? []}
                />

                <div className="flex flex-wrap gap-3 text-sm">
                  <Link href={{ pathname: '/compare', query: { search: outName } }} className="rounded-md border px-4 py-2 hover:bg-muted">View in model library</Link>
                  <Link href={{ pathname: '/create-test', query: { provider: 'transformers', model: run.out_dir } }} className="rounded-md border px-4 py-2 hover:bg-muted">Evaluate this model</Link>
                  <Link href={{ pathname: '/weights', query: { kind: 'compare', source_model: run.source_model, modified_model: run.out_dir } }} className="rounded-md border px-4 py-2 hover:bg-muted">Compare behavior with original</Link>
                </div>
              </>
            ) : (
              <p className="rounded-lg border bg-card p-4 text-sm text-muted-foreground">
                This run kept only its adapter, under <code className="text-xs">{summary.adapter_path ?? run.out_dir}</code>.
                Merge is off, so no model directory was written and nothing new appears in the library.
              </p>
            )}

            {summary.manifest && Object.keys(summary.manifest).length > 0 && <ManifestCard manifest={summary.manifest} />}
            {modelDetail && <ProvenanceCard detail={modelDetail} />}

            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <h2 className="mb-2 text-lg font-semibold">Use it in a test run</h2>
              <p className="mb-3 text-sm text-muted-foreground">
                Serve it through the transformers provider. Run the baseline through the same
                provider too — comparing this against a stock model served through Ollama would
                measure the serving stack as much as the weights.
              </p>
              <p className="text-sm text-muted-foreground">Evaluate this model opens a test with the saved checkpoint already selected. Layer-by-layer comparison shows changes in activations; behavior comparison measures responses and capability on held-out prompts.</p>
            </div>
          </>
        )}
      </div>
    </Layout>
  )
}

function TrainingSection({ run, summary }: { run: { kind: string; method: string | null }; summary: Record<string, any> }) {
  const train = summary.train ?? {}
  const distill = summary.distill ?? summary.manifest?.extra?.distill
  const fmt = (value: any, digits = 4) => (typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—')
  return (
    <>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard title="Optimizer steps" value={train.steps ?? '—'} subtitle={train.trainable_params != null ? `${Number(train.trainable_params).toLocaleString()} trainable parameters` : undefined} />
        <MetricCard title="Final train loss" value={fmt(train.final_loss)} subtitle={train.mean_kl != null ? `mean KL to teacher ${fmt(train.mean_kl)}` : 'response tokens only'} />
        <MetricCard
          title="Eval loss"
          value={fmt(train.eval_loss_after)}
          subtitle={train.eval_loss_before != null ? `from ${fmt(train.eval_loss_before)} before training` : 'no held-out rows'}
        />
        <MetricCard
          title={distill ? 'Teacher' : 'Adapter'}
          value={distill ? String(distill.teacher_model ?? '—').split('/').pop() ?? '—' : summary.merged ? 'merged' : 'kept separately'}
          subtitle={distill ? `${distill.level} level${distill.level === 'logit' ? `, T=${distill.temperature}` : ''}` : train.target_modules ? `${train.target_modules.length} module names` : undefined}
        />
      </div>

      {Array.isArray(summary.history) && summary.history.length > 0 && <TrainingCurve history={summary.history} />}

      {Array.isArray(summary.teacher_samples) && summary.teacher_samples.length > 0 && (
        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h2 className="text-lg font-semibold">What the teacher said</h2>
          <p className="mb-3 mt-1 text-sm text-muted-foreground">
            The first few responses the student was trained to imitate. The full set is in the run
            directory as <code className="text-xs">teacher_responses.jsonl</code>.
          </p>
          <ul className="space-y-3 text-sm">
            {summary.teacher_samples.map((sample: { prompt: string; response: string }, i: number) => (
              <li key={i} className="rounded border p-3">
                <p className="font-medium">{sample.prompt}</p>
                <p className="mt-1 whitespace-pre-wrap text-muted-foreground">{sample.response}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {summary.worker_log && (
        <p className="text-xs text-muted-foreground">
          Trained in a worker process; its log is <code>{summary.worker_log}</code>.
          {run.method ? ` Method: ${run.method}.` : ''}
        </p>
      )}
    </>
  )
}

function RoutingSection({
  runId,
  sourceModel,
  summary,
}: {
  runId: number
  sourceModel: string
  summary: Record<string, any>
}) {
  const { data, isLoading, error } = useRoutingStats(runId)
  const top = summary.ranking?.[0]
  const consistent = summary.consistency?.gate_vs_expert_counts_match !== false
  return (
    <>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard title="MoE layers" value={summary.moe_layers ?? '—'} subtitle={`of ${summary.n_layers ?? '?'} decoder layers`} />
        <MetricCard title="Experts per token" value={summary.top_k ?? '—'} subtitle={summary.model_type ?? undefined} />
        <MetricCard
          title="Most differential"
          value={top ? `L${top.layer} E${top.expert}` : '—'}
          subtitle={top ? `${(top.delta * 100).toFixed(1)} pts more on harmful` : 'no ranking'}
        />
        <MetricCard
          title="Replay verified"
          value={consistent ? 'yes' : 'no'}
          subtitle={consistent ? 'gate replay matched expert row counts' : 'fractions are unverified'}
        />
      </div>
      {isLoading && <LoadingSpinner />}
      {error && (
        <p className="text-sm text-destructive">
          Could not load the per-expert tables: {error instanceof Error ? error.message : 'unknown error'}
        </p>
      )}
      {data && <RoutingHeatmap result={data} sourceModel={sourceModel} />}
      <p className="text-sm text-muted-foreground">
        Nothing here was written to a model. Pick experts in the table and edit exactly those; the
        result is a normal model directory whose manifest says the edit was partial.
      </p>
    </>
  )
}


/**
 * The loop's own record: what it tried, what it kept, and whether the bytes on
 * disk passed. Rendered for failed runs too, since a search that found nothing
 * admissible is a result, and the trial log is the whole point of it.
 */
function AutotuneSection({ summary }: { summary: any }) {
  const winner = summary.winner ?? null
  const verification = summary.verification ?? null
  const spec = summary.spec ?? {}
  return (
    <>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricCard
          title="Kept"
          value={winner ? `rank ${winner.rank}, k ${Number(winner.k).toFixed(2)}` : 'nothing'}
          subtitle={
            winner
              ? `embeddings ${winner.include_embeddings ? 'edited' : 'untouched'} · refuses ${(winner.refuse_harmful * 100).toFixed(0)}%`
              : 'no candidate cleared every gate'
          }
        />
        <MetricCard
          title="Candidates tried"
          value={`${summary.candidates_tried ?? 0} / ${summary.candidates_planned ?? '—'}`}
          subtitle={summary.stopped_early ? 'stopped at the first that met the target' : 'scored every candidate, then picked'}
        />
        <MetricCard
          title="Verified from disk"
          value={verification ? (verification.passed ? 'yes' : 'NO') : '—'}
          subtitle={verification ? `${verification.n_prompts} prompts, ${verification.sampled ? 'greedy + sampled' : 'greedy only'}` : 'not reached'}
        />
        <MetricCard
          title="Refusal target"
          value={summary.target_met ? 'met' : 'not met'}
          subtitle={`≤ ${((spec.max_refusal ?? 0.1) * 100).toFixed(0)}% on held-out prompts`}
        />
      </div>

      {summary.embeddings_tied && (
        <p className="rounded-lg border bg-card p-4 text-sm text-muted-foreground">
          This model ties its output layer to the embedding table, so candidates that edit the
          embeddings also rewrite the unembedding. The loop tried the untouched-embeddings edit
          first for that reason.
        </p>
      )}

      <FrontierChart
        baseline={summary.baseline?.greedy}
        frontier={summary.trials ?? []}
        best={winner}
        factualFloor={spec.factual_floor ?? 0.05}
      />

      <TrialLog trials={summary.trials ?? []} winner={winner} maxRefusal={spec.max_refusal} />

      {verification && <VerificationCard report={verification} />}
    </>
  )
}
