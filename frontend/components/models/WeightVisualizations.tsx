import { useId, useMemo, useState } from 'react'
import Link from 'next/link'

import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { CompareTable } from '@/components/weights/CompareTable'
import { FrontierChart } from '@/components/weights/FrontierChart'
import { LayerScoreChart } from '@/components/weights/LayerScoreChart'
import { RefusalTimelineChart } from '@/components/weights/RefusalTimelineChart'
import { SubspaceCurveChart } from '@/components/weights/SubspaceCurveChart'
import { SweepTable } from '@/components/weights/SweepTable'
import { ProbeResults } from '@/components/weights/ProbeResults'
import type { WeightRunKind } from '@/lib/api'
import { useWeightRun, useWeightRuns } from '@/lib/hooks'
import { formatApiError, formatDate } from '@/lib/utils'

interface WeightVisualizationsProps {
  /** Undefined shows all models; an empty list deliberately shows no models. */
  modelRefs?: string[]
  displayNames?: Record<string, string>
}

const RUN_LABELS: Record<WeightRunKind, string> = {
  direction: 'Direction',
  sweep: 'Causal check',
  select: 'Capability search',
  autotune: 'Verified edit',
  surgery: 'Weight surgery',
  compare: 'Behavior comparison',
  probe: 'Intent probe',
  routing: 'Expert routing',
  expert_surgery: 'Expert edit',
  lora: 'LoRA fine-tune',
  distill: 'Distillation',
  induce: 'Targeted refusal',
  hneurons: 'Hallucination neurons',
  hneuron_bake: 'H-neuron bake',
  redteam: 'Red-team',
  embed_align: 'Embedding alignment',
  embed_extract: 'Embedding extraction',
  embed_recon: 'Embedding reconstruction',
}

const hasRows = (value: unknown): boolean => Array.isArray(value) && value.length > 0

export function WeightVisualizations({ modelRefs, displayNames = {} }: WeightVisualizationsProps) {
  const selectId = useId()
  const { data: allRuns = [], isLoading, error, refetch } = useWeightRuns({ limit: 1000, status: 'completed' })
  const runs = useMemo(() => allRuns.filter(run =>
    run.status === 'completed' && ['direction', 'sweep', 'select', 'compare', 'probe'].includes(run.kind) &&
    (modelRefs === undefined || modelRefs.includes(run.source_model) ||
      modelRefs.includes(run.metadata?.modified_model)),
  ).sort((a, b) => b.id - a.id), [allRuns, modelRefs])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const selectedRunId = runs.find(run => run.id === selectedId)?.id ?? runs[0]?.id ?? 0
  const { data: run, isLoading: loadingRun, error: runError, refetch: refetchRun } = useWeightRun(selectedRunId)
  const summary = run?.metadata?.summary ?? {}
  const directionScores = run?.kind === 'direction' && hasRows(summary.layer_scores)
  const sweepCurve = run?.kind === 'sweep' && hasRows(summary.curve)
  const sweepTimelines = run?.kind === 'sweep' && hasRows(summary.timelines)
  const sweepRows = run?.kind === 'sweep' && hasRows(summary.rows)
  const frontier = run?.kind === 'select' && hasRows(summary.frontier)
  const comparison = run?.kind === 'compare' && summary.metrics?.baseline &&
    summary.metrics?.modified && summary.deltas
  const probe = run?.kind === 'probe'
  const hasPlots = directionScores || sweepCurve || sweepTimelines || sweepRows || frontier || comparison || probe

  return (
    <section aria-label="Weight experiment plots" className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold">Weight experiment plots</h2>
        <p className="text-sm text-muted-foreground">
          Explore saved layer scores, causal checks, capability tradeoffs, and behavior comparisons.
          Results come from completed runs on the connected server.
        </p>
      </div>

      {isLoading ? (
        <div role="status" className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
          <LoadingSpinner size="sm" /> Loading weight experiments…
        </div>
      ) : error ? (
        <div role="alert" className="rounded-lg border border-destructive/40 p-4 text-sm">
          <p>Could not load weight experiments: {formatApiError(error)}</p>
          <button type="button" onClick={() => refetch()} className="mt-2 underline">Try again</button>
        </div>
      ) : !runs.length ? (
        <div className="rounded-lg border bg-muted/20 p-6 text-sm">
          <p className="font-medium">No completed weight experiments in this view</p>
          <p className="mt-1 text-muted-foreground">
            {modelRefs !== undefined
              ? 'Choose another model or experiment, or show all models to find saved plots.'
              : 'Saved results belong to the connected server. Runs recorded on another server will not appear here.'}
            {' '}Direction, causal check, capability search, and comparison runs produce results here.
          </p>
          <Link href="/weights" className="mt-3 inline-block underline">Open weight experiments</Link>
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-3 rounded-lg border bg-card p-4">
            <div className="min-w-0 flex-1 basis-80">
              <label htmlFor={selectId} className="mb-1 block text-sm font-medium">Saved weight experiment</label>
              <select
                id={selectId}
                value={selectedRunId}
                onChange={event => setSelectedId(Number(event.target.value))}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
              >
                {runs.map(item => (
                  <option key={item.id} value={item.id}>
                    #{item.id} · {RUN_LABELS[item.kind]} · {displayNames[item.source_model] || item.source_model}
                    {item.completed_at || item.created_at ? ` · ${formatDate(item.completed_at || item.created_at)}` : ''}
                  </option>
                ))}
              </select>
            </div>
            <Link href={`/weights/${selectedRunId}`} className="rounded-md border px-3 py-2 text-sm hover:bg-muted">
              Open full weight results
            </Link>
          </div>

          {loadingRun ? (
            <div role="status" className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
              <LoadingSpinner size="sm" /> Loading saved plots…
            </div>
          ) : runError ? (
            <div role="alert" className="rounded-lg border border-destructive/40 p-4 text-sm">
              <p>Could not load this experiment: {formatApiError(runError)}</p>
              <button type="button" onClick={() => refetchRun()} className="mt-2 underline">Try again</button>
            </div>
          ) : !hasPlots ? (
            <p className="rounded-lg border bg-muted/20 p-5 text-sm text-muted-foreground">
              This run has no saved chart data on the connected server. Open its full results to inspect the recorded summary, or choose another experiment.
            </p>
          ) : (
            <div className="space-y-5">
              {directionScores && <LayerScoreChart layerScores={summary.layer_scores} selectedLayer={summary.layer} minUsableAuc={summary.min_usable_auc ?? 0.9} />}
              {sweepCurve && <SubspaceCurveChart rows={summary.curve} summary={summary} />}
              {sweepTimelines && <RefusalTimelineChart timelines={summary.timelines} />}
              {sweepRows && <SweepTable rows={summary.rows} verdict={summary.verdict} ablateDeltaPoints={summary.ablate_delta_points} factualDeltaPoints={summary.factual_delta_points} layer={summary.layer} />}
              {frontier && <FrontierChart baseline={summary.baseline} frontier={summary.frontier} best={summary.best} factualFloor={summary.factual_floor ?? 0.05} />}
              {comparison && <CompareTable metrics={summary.metrics} deltas={summary.deltas} responses={summary.responses} prompts={summary.prompts} />}
              {probe && <ProbeResults summary={summary} />}
            </div>
          )}
        </>
      )}
    </section>
  )
}
