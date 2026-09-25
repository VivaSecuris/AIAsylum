import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { CurveRow } from '@/lib/api'

interface Props {
  rows: CurveRow[]
  summary: {
    k50_rank?: number | null
    max_compliance?: number | null
    rank_at_max?: number | null
    monotone?: boolean | null
    baseline_refusal?: number | null
    rank?: number
    weights?: number[] | null
    method?: string
    any_degenerate?: boolean
  }
}

const pct = (v: number) => `${(v * 100).toFixed(0)}%`

/**
 * Refusal against the number of subspace directions removed.
 *
 * The question this answers is "how many directions does this model need".
 * One difference-in-means vector is enough on small models; on Qwen3-8B the
 * RFM-AGOP paper needs three before compliance passes half. The dashed line
 * is that half-way mark, and the first rank to cross it is called out.
 *
 * Degenerate rows are drawn hollow and excluded from the summary: a model that
 * has collapsed into repetition emits no refusal phrases, which a phrase
 * detector scores as perfect compliance.
 */
export function SubspaceCurveChart({ rows, summary }: Props) {
  if (!rows?.length) {
    return <p className="text-sm text-muted-foreground">This curve recorded no rows.</p>
  }
  const firstK = rows.find((r) => r.rank > 0)?.k
  const data = rows
    .filter((r) => r.rank === 0 || r.k === firstK)
    .map((r) => ({
      rank: r.rank,
      refusal: r.degenerate ? null : r.refusal_rate,
      refusal_degenerate: r.degenerate ? r.refusal_rate : null,
      factual: r.factual_acc ?? null,
      label: r.label,
      degenerate: r.degenerate,
    }))

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <div className="mb-4">
        <h2 className="text-lg font-semibold">How many directions refusal needs</h2>
        <p className="text-sm text-muted-foreground">
          Rank 0 is the unedited model. Each further point removes one more direction of the{' '}
          {summary.method === 'rfm_agop' ? 'RFM-AGOP cone, weighted by eigenvalue' : 'subspace'}, at
          inference time. Nothing was written to disk.
        </p>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 16, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="rank" type="number" domain={[0, 'dataMax']} allowDecimals={false}
                 label={{ value: 'directions removed', position: 'insideBottom', offset: -4 }} />
          <YAxis domain={[0, 1]} tickFormatter={pct} />
          <Tooltip
            content={({ active, payload }: any) => {
              if (!active || !payload?.length) return null
              const p = payload[0].payload
              return (
                <div className="rounded-md border bg-background px-3 py-2 text-xs shadow-sm">
                  <div className="font-medium">{p.label}</div>
                  <div>refusal {pct(p.refusal ?? p.refusal_degenerate ?? 0)}{p.degenerate ? ' (degenerate)' : ''}</div>
                  {p.factual != null && <div>factual {pct(p.factual)}</div>}
                </div>
              )
            }}
          />
          <Legend />
          <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="3 3"
                         label={{ value: 'half compliance', position: 'insideTopRight', fontSize: 11 }} />
          {summary.k50_rank != null && (
            <ReferenceLine x={summary.k50_rank} stroke="#10b981" strokeWidth={2}
                           label={{ value: `rank ${summary.k50_rank} crosses`, position: 'top', fontSize: 11 }} />
          )}
          <Line type="monotone" dataKey="refusal" name="refusal" stroke="#3b82f6" strokeWidth={2} connectNulls />
          <Line type="monotone" dataKey="refusal_degenerate" name="refusal (degenerate)" stroke="#f59e0b"
                strokeWidth={0} dot={{ r: 5, fill: 'transparent', stroke: '#f59e0b' }} />
          <Line type="monotone" dataKey="factual" name="factual accuracy" stroke="#8b5cf6" strokeWidth={2}
                strokeDasharray="4 3" connectNulls />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
        <Stat label="Smallest rank reaching half compliance"
              value={summary.k50_rank != null ? String(summary.k50_rank) : 'none'} />
        <Stat label="Best compliance"
              value={summary.max_compliance != null
                ? `${pct(summary.max_compliance)} at rank ${summary.rank_at_max}` : '—'} />
        <Stat label="Monotone" value={summary.monotone == null ? '—' : summary.monotone ? 'yes' : 'no'} />
      </div>

      {summary.weights && summary.weights.length > 1 && (
        <p className="mt-3 text-xs text-muted-foreground">
          Removal weights (μᵢ/μ₁): {summary.weights.map((w) => w.toFixed(3)).join(', ')}. The first direction is
          removed in full; the rest in proportion to how much refusal they carry.
        </p>
      )}
      {summary.any_degenerate && (
        <p className="mt-2 text-xs text-amber-700 dark:text-amber-300">
          At least one configuration collapsed into repetition; its refusal number is hollow above and left out
          of the summary.
        </p>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-muted/40 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  )
}
