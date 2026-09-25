import { useInterpArtifact } from '@/lib/hooks'
import { formatApiError } from '@/lib/utils'

interface PatchResult {
  layer: number
  position: number
  direction: string
  recovered: number
  recovered_kl: number
  top_flipped: boolean
  top_target_after: string[]
}

interface PatchingPayload {
  enabled?: boolean
  claim?: string
  forward_passes?: number
  notes?: string[]
  summary?: Record<string, {
    best_layer: number
    best_component: string
    best_component_index?: number | null
    best_recovered: number
    best_top_flipped: boolean
    same_top_token: boolean
    beats_null?: boolean
  }>
  baseline?: Record<string, { recovered: number }>
  experiments?: Array<{ id: string; description: string; results: PatchResult[] }>
}

const percentage = (value: number | null | undefined) =>
  typeof value === 'number' && Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : 'Not measured'

const directionLabel = (direction: string) =>
  direction === 'A_to_B' ? 'A activations into prompt B' : direction === 'B_to_A' ? 'B activations into prompt A' : direction

export function PatchingPanel({ runId }: { runId: number }) {
  const { data, isLoading, error } = useInterpArtifact(runId, 'patching_results.json')
  const payload = data as PatchingPayload | undefined
  if (isLoading) return <p className="text-sm text-muted-foreground">Loading activation patching…</p>
  if (error) return <p role="alert" className="text-sm text-destructive">{formatApiError(error, 'Could not load activation patching')}</p>
  if (!payload) return null
  const experiments = payload.experiments ?? []

  return (
    <div className="space-y-4">
      <div className="rounded-lg border bg-card p-5">
        <h2 className="text-lg font-semibold">Activation patching</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Replaces a captured activation from one prompt in the other prompt’s computation,
          then runs the model again. Recovery measures how far the prediction moves toward the
          source: 0% means no recovery, 100% closes the original gap, and values outside this
          range indicate movement away from the source or an overshoot.
        </p>
        <p className="mt-2 text-xs text-muted-foreground">
          {payload.forward_passes ?? 'Unknown number of'} forward passes. Effects apply to this
          prompt pair and selected token positions. A single random perturbation is a comparison
          baseline, not a statistical significance test. Test additional prompts before generalizing.
        </p>
        {payload.notes?.map((note, i) => <p key={i} className="mt-2 text-xs text-amber-700 dark:text-amber-300">{note}</p>)}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {Object.entries(payload.summary ?? {}).map(([direction, summary]) => (
          <div key={direction} className="rounded-lg border bg-card p-5">
            <h3 className="text-sm font-semibold">{directionLabel(direction)}</h3>
            <p className="mt-1 text-xs text-muted-foreground">Best tested {summary.best_component} at hidden-state index {summary.best_layer}{summary.best_component_index != null ? `, component ${summary.best_component_index}` : ''}</p>
            <dl className="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div><dt className="text-xs text-muted-foreground">Gap recovered</dt><dd className="text-xl font-semibold">{percentage(summary.best_recovered)}</dd></div>
              <div><dt className="text-xs text-muted-foreground">Random perturbation</dt><dd className="text-xl font-semibold">{percentage(payload.baseline?.[direction]?.recovered)}</dd></div>
              <div><dt className="text-xs text-muted-foreground">Top prediction changed to source</dt><dd>{summary.same_top_token ? 'Already matched' : summary.best_top_flipped ? 'Yes' : 'No'}</dd></div>
              <div><dt className="text-xs text-muted-foreground">Exceeds random baseline by &gt;5 points</dt><dd>{summary.beats_null == null ? 'Not measured' : summary.beats_null ? 'Yes' : 'No'}</dd></div>
            </dl>
            <p className="mt-3 text-xs text-muted-foreground">{summary.same_top_token ? 'The original top tokens match, so recovery uses the KL distribution gap.' : 'Recovery uses the logit margin between the two original top tokens.'}</p>
          </div>
        ))}
      </div>

      <details className="rounded-lg border bg-card p-4">
        <summary className="cursor-pointer text-sm font-medium">All patch experiments ({experiments.length})</summary>
        <p className="mt-2 text-xs text-muted-foreground">Hidden-state index 0 is the embedding; the final index is the final residual. Token positions are relative to the aligned analysis window.</p>
        <div className="mt-3 max-h-[480px] overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 border-b bg-card text-muted-foreground"><tr><th className="p-2">Experiment</th><th className="p-2">Position</th><th className="p-2">Recovery</th><th className="p-2">KL recovery</th><th className="p-2">New top token</th></tr></thead>
            <tbody>{experiments.flatMap((experiment) => (experiment.results ?? []).map((result, i) => (
              <tr key={`${experiment.id}-${i}`} className="border-b last:border-0"><td className="p-2">{experiment.description}</td><td className="p-2">{result.position}</td><td className="p-2">{percentage(result.recovered)}</td><td className="p-2">{percentage(result.recovered_kl)}</td><td className="whitespace-pre-wrap p-2 font-mono">{JSON.stringify(result.top_target_after?.[0] ?? '')}</td></tr>
            )))}</tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
