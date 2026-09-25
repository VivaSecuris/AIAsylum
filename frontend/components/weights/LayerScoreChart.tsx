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
import { useState } from 'react'

import type { LayerScore } from '@/lib/api'

interface Props {
  layerScores: LayerScore[]
  selectedLayer?: number | null
  minUsableAuc?: number
}

/**
 * How well each layer separates the two classes.
 *
 * Two series on two axes, because they answer different questions and are on
 * incompatible scales. AUC saturates at 1.0 across many layers on a cleanly
 * separable pair, which is exactly why the engine breaks ties on Cohen's d --
 * so on a saturated run the dashed d line is the entire story and the solid
 * AUC line carries almost none.
 *
 * The AUC axis floors at 0.5 rather than 0: 0.5 is chance, and anything below
 * it is a direction pointing the other way, not a worse one. Starting at zero
 * would waste half the plot on impossible values.
 */
export function LayerScoreChart({ layerScores, selectedLayer, minUsableAuc = 0.9 }: Props) {
  const [showCohensD, setShowCohensD] = useState(true)
  const [showStableRank, setShowStableRank] = useState(false)
  const hasStableRank = layerScores?.some((s) => s.stable_rank != null) ?? false

  if (!layerScores?.length) {
    return (
      <p className="text-sm text-muted-foreground">
        No per-layer scores were recorded for this run.
      </p>
    )
  }

  const data = layerScores.map((s) => ({
    layer: s.layer,
    auc: s.auc,
    cohens_d: s.cohens_d,
    stable_rank: s.stable_rank ?? null,
  }))

  const top5 = [...layerScores].sort((a, b) => b.auc - a.auc).slice(0, 5)

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">Separation by layer</h2>
          <p className="text-sm text-muted-foreground">
            Measured on the held-out half. The layer was chosen here, not on the data the
            direction was fitted to.
          </p>
        </div>
        <div className="flex shrink-0 flex-col gap-1 text-xs text-muted-foreground">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showCohensD}
              onChange={(e) => setShowCohensD(e.target.checked)}
            />
            Show Cohen&apos;s d
          </label>
          {hasStableRank && (
            <label className="flex items-center gap-2" title="Stable rank of the benign-centred refusal residuals; higher predicts weaker single-vector ablation (arXiv 2608.25390)">
              <input
                type="checkbox"
                checked={showStableRank}
                onChange={(e) => setShowStableRank(e.target.checked)}
              />
              Show stable rank
            </label>
          )}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 16, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="layer"
            type="number"
            domain={['dataMin', 'dataMax']}
            label={{ value: 'layer', position: 'insideBottom', offset: -4 }}
          />
          <YAxis
            yAxisId="auc"
            orientation="left"
            domain={[0.5, 1]}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          {showCohensD && <YAxis yAxisId="d" orientation="right" domain={[0, 'auto']} />}
          {showStableRank && <YAxis yAxisId="sr" orientation="right" domain={[0, 'auto']} />}
          <Tooltip
            content={({ active, payload }: any) => {
              if (!active || !payload?.length) return null
              const p = payload[0].payload
              return (
                <div className="rounded-md border bg-background px-3 py-2 text-xs shadow-sm">
                  layer {p.layer} — AUC {p.auc?.toFixed(3)} · d {p.cohens_d?.toFixed(2)}
                  {p.stable_rank != null && <> · stable rank {p.stable_rank.toFixed(1)}</>}
                </div>
              )
            }}
          />
          <Legend />

          {/* Chance. A direction at 0.5 separates nothing. */}
          <ReferenceLine yAxisId="auc" y={0.5} stroke="#94a3b8" strokeDasharray="2 2" />
          {/* The gate the pipeline actually enforces. */}
          <ReferenceLine
            yAxisId="auc"
            y={minUsableAuc}
            stroke="#f59e0b"
            strokeDasharray="3 3"
            label={{ value: `usable ≥ ${minUsableAuc.toFixed(2)}`, position: 'insideBottomLeft', fontSize: 11 }}
          />
          {/* The most informative mark here: it shows whether the chosen layer
              was a genuine spike or a Cohen's-d tiebreak on a flat plateau. */}
          {selectedLayer != null && (
            <ReferenceLine
              yAxisId="auc"
              x={selectedLayer}
              stroke="#10b981"
              strokeWidth={2}
              label={{ value: `selected L${selectedLayer}`, position: 'top', fontSize: 11 }}
            />
          )}

          <Line
            yAxisId="auc"
            type="monotone"
            dataKey="auc"
            name="held-out AUC"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
          />
          {showCohensD && (
            <Line
              yAxisId="d"
              type="monotone"
              dataKey="cohens_d"
              name="Cohen's d"
              stroke="#8b5cf6"
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={false}
            />
          )}
          {showStableRank && (
            <Line
              yAxisId="sr"
              type="monotone"
              dataKey="stable_rank"
              name="stable rank of refusal residuals"
              stroke="#f97316"
              strokeWidth={2}
              strokeDasharray="2 2"
              dot={false}
              connectNulls
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-4">
        <h3 className="mb-2 text-sm font-medium">Top layers by held-out separation</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-1 font-medium">layer</th>
              <th className="py-1 font-medium">AUC</th>
              <th className="py-1 font-medium">Cohen&apos;s d</th>
              {hasStableRank && <th className="py-1 font-medium">stable rank</th>}
            </tr>
          </thead>
          <tbody>
            {top5.map((s) => (
              <tr
                key={s.layer}
                className={s.layer === selectedLayer ? 'font-medium text-foreground' : ''}
              >
                <td className="py-1 font-mono">
                  {s.layer}
                  {s.layer === selectedLayer && (
                    <span className="ml-2 rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200">
                      selected
                    </span>
                  )}
                </td>
                <td className="py-1 font-mono">{s.auc.toFixed(4)}</td>
                <td className="py-1 font-mono">{s.cohens_d.toFixed(2)}</td>
                {hasStableRank && (
                  <td className="py-1 font-mono">{s.stable_rank != null ? s.stable_rank.toFixed(1) : '—'}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
