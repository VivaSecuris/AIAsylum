import { AlertTriangle } from 'lucide-react'

import { useInterpArtifact } from '@/lib/hooks'

interface Props {
  runId: number
}

/**
 * Circuit findings, rendered natively rather than left inside the dashboard.
 *
 * The honesty notice is not decoration. These cards score heads and neurons by
 * how much they differ between two runs, using fixed thresholds (0.1 for
 * heads, 0.05 for neurons, 0.2 for activation contrast) that were never
 * validated against held-out data -- unlike the direction pipeline, which gates
 * on held-out AUC. "Safety neurons" are named from a single contrast pair, and
 * the project's own methodology notes say a single-direction result is
 * consistent with mere correlation. The minimal circuit below is the causal
 * one, and the two must not be read as equally strong.
 */
export function CircuitPanel({ runId }: Props) {
  const { data: circuit, isLoading } = useInterpArtifact(runId, 'circuit_payload.json')
  const { data: minimal } = useInterpArtifact(runId, 'minimal_circuit_payload.json')

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading circuit data…</p>

  if (!circuit && !minimal) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-sm text-muted-foreground">
        This run produced no circuit data. Component analysis needs attention or MLP capture,
        and the minimal circuit needs its own flag — all are off by default because each one
        multiplies capture cost or search time.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {circuit && (
        <>
          <div className="flex items-start gap-2 rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 text-sm dark:bg-yellow-950">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-600" />
            <div>
              <p className="font-medium">Correlational, from one prompt pair</p>
              <p className="text-muted-foreground">
                These components are ranked by how much they differ between the two runs, using
                fixed thresholds that were not validated on held-out data. That is a lead worth
                following, not evidence that these components implement the behaviour. The
                minimal circuit, if present below, is the causal result.
              </p>
            </div>
          </div>

          <CircuitCard title="At the last token" card={circuit.spike_layer_last_token} />
          <CircuitCard title="At the first divergence" card={circuit.spike_layer_first_divergence} />
          <SafetyNeurons clusters={circuit.safety_neuron_clusters} />
        </>
      )}

      {minimal?.circuit && <MinimalCircuit payload={minimal} />}
    </div>
  )
}

function CircuitCard({ title, card }: { title: string; card: any }) {
  if (!card) return null

  const heads = card.involved_heads ?? []
  const neurons = card.involved_neurons ?? []
  const maxHead = Math.max(...heads.map((h: any) => h.contribution ?? 0), 1e-9)
  const maxNeuron = Math.max(...neurons.map((n: any) => n.contribution ?? 0), 1e-9)

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        <span className="font-mono text-xs text-muted-foreground">
          layer {card.layer} · token {card.token} · {card.total_components} components
        </span>
      </div>

      {card.circuit_summary && (
        <p className="mb-4 text-sm text-muted-foreground">{card.circuit_summary}</p>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        <Bars
          label="Attention heads"
          empty={card.head_available ? 'None above threshold.' : 'Attention capture was off.'}
          rows={heads.map((h: any) => ({
            key: `H${h.head}`,
            value: h.contribution,
            max: maxHead,
          }))}
          color="bg-blue-500"
        />
        <Bars
          label="MLP neurons"
          empty={card.neuron_available ? 'None above threshold.' : 'MLP capture was off.'}
          rows={neurons.slice(0, 20).map((n: any) => ({
            key: `#${n.neuron}`,
            value: n.contribution,
            max: maxNeuron,
          }))}
          color="bg-violet-500"
        />
      </div>
    </div>
  )
}

function SafetyNeurons({ clusters }: { clusters: any }) {
  if (!clusters?.mlp_available) return null
  const neurons = (clusters.safety_neurons ?? []).slice(0, 20)
  const max = Math.max(...neurons.map((n: any) => n.activation_contrast ?? 0), 1e-9)

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <h3 className="text-sm font-semibold">High-contrast neurons</h3>
      <p className="mb-4 mt-1 text-xs text-muted-foreground">
        Neurons whose mean activation differs most between the two runs —{' '}
        {clusters.num_safety_neurons} of {clusters.total_neurons} (
        {Number(clusters.safety_neuron_percentage ?? 0).toFixed(1)}%). The engine calls these
        &quot;safety neurons&quot;; from a single prompt pair, high contrast is the most that
        can honestly be claimed.
      </p>
      <Bars
        label=""
        empty="None above threshold."
        rows={neurons.map((n: any) => ({
          key: `#${n.neuron}`,
          value: n.activation_contrast,
          max,
        }))}
        color="bg-amber-500"
      />
    </div>
  )
}

/** Components arrive as `(layer, "head" | "neuron", index)` tuples. */
function componentLabel(c: any): string {
  if (typeof c === 'string') return c
  if (Array.isArray(c) && c.length >= 3) return `L${c[0]} ${c[1]} ${c[2]}`
  if (Array.isArray(c)) return c.join(' ')
  return JSON.stringify(c)
}

function MinimalCircuit({ payload }: { payload: any }) {
  const circuit = payload.circuit ?? []
  const history = payload.metric_history ?? []
  return (
    <div className="rounded-lg border border-emerald-500/50 bg-card p-6 shadow-sm">
      <h3 className="text-sm font-semibold">Minimal sufficient circuit</h3>
      <p className="mb-4 mt-1 text-xs text-muted-foreground">
        Found by scrubbing components away and keeping only those the behaviour needs. This is
        a causal claim, unlike the contribution rankings above. Only attention heads are
        searched: the pre-MLP capture hook is unimplemented, so neurons are never candidates.
      </p>

      <div className="flex flex-wrap gap-2">
        {circuit.length === 0 && (
          <span className="text-xs text-muted-foreground">
            The search kept no components.
          </span>
        )}
        {circuit.map((c: any, i: number) => (
          <span
            key={i}
            className="rounded bg-emerald-100 px-2 py-1 font-mono text-xs text-emerald-900 dark:bg-emerald-950 dark:text-emerald-100"
          >
            {componentLabel(c)}
          </span>
        ))}
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
        <Stat label="components kept" value={circuit.length} />
        <Stat label="candidates searched" value={payload.num_candidates} />
        <Stat
          label={`final ${payload.metric ?? 'divergence'}`}
          value={
            payload.final_metric != null ? Number(payload.final_metric).toFixed(4) : undefined
          }
        />
      </dl>

      {history.length > 1 && (
        <p className="mt-3 text-xs text-muted-foreground">
          The search drove {payload.metric ?? 'the metric'} from{' '}
          <span className="font-mono">{Number(history[0]).toFixed(4)}</span> to{' '}
          <span className="font-mono">
            {Number(history[history.length - 1]).toFixed(4)}
          </span>{' '}
          over {history.length} steps. Lower is a closer match to the unscrubbed run.
        </p>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono">{value ?? '—'}</dd>
    </div>
  )
}

function Bars({
  label,
  rows,
  color,
  empty,
}: {
  label: string
  rows: Array<{ key: string; value: number; max: number }>
  color: string
  empty: string
}) {
  return (
    <div>
      {label && <p className="mb-2 text-xs font-medium">{label}</p>}
      {rows.length === 0 ? (
        <p className="text-xs text-muted-foreground">{empty}</p>
      ) : (
        <div className="space-y-1">
          {rows.map((r) => (
            <div key={r.key} className="flex items-center gap-2 text-xs">
              <span className="w-14 shrink-0 font-mono text-muted-foreground">{r.key}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                <div
                  className={`h-full rounded-full ${color}`}
                  style={{ width: `${Math.min(100, (r.value / r.max) * 100)}%` }}
                />
              </div>
              <span className="w-12 shrink-0 text-right font-mono text-muted-foreground">
                {r.value?.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
