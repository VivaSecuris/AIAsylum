import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'

interface FrontierRow {
  rank: number
  k: number
  refuse_harmful: number
  factual_acc: number
  factual_drop: number
  degenerate: boolean
  accepted: boolean
}

interface Props {
  baseline?: { refuse_harmful: number; factual_acc: number }
  frontier: FrontierRow[]
  best?: FrontierRow | null
  factualFloor?: number
}

const pct = (v: number) => `${(v * 100).toFixed(0)}%`

/**
 * What more compliance actually costs.
 *
 * Every other view in this section reports one configuration. This one shows
 * the trade, which is the only way to see that the last few points of
 * compliance are bought with capability rather than earned.
 *
 * The vertical line is the capability floor: baseline factual accuracy minus
 * the allowed drop. Everything left of it was rejected no matter how compliant
 * it looked, which is the whole point of running the control -- a model that
 * refuses nothing because it has been broken scores the same as one that was
 * cleanly ablated.
 */
export function FrontierChart({ baseline, frontier, best, factualFloor = 0.05 }: Props) {
  if (!frontier?.length) {
    return <p className="text-sm text-muted-foreground">This search scored no candidates.</p>
  }

  const floor = baseline ? baseline.factual_acc - factualFloor : undefined
  const mark = (r: FrontierRow) => ({ ...r, x: r.factual_acc, y: r.refuse_harmful })

  const accepted = frontier.filter((r) => r.accepted).map(mark)
  const rejected = frontier.filter((r) => !r.accepted && !r.degenerate).map(mark)
  const degenerate = frontier.filter((r) => r.degenerate).map(mark)
  const bestPoint = best ? [mark(best as FrontierRow)] : []

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <h2 className="text-lg font-semibold">What compliance costs</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        Each point is one subspace rank at one removal strength, previewed at inference time.
        Up is more compliant; right is more capable. Nothing here was written to disk.
      </p>

      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 16, right: 24, bottom: 16, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name="factual accuracy"
            domain={[0, 1]}
            tickFormatter={pct}
            label={{ value: 'factual accuracy (capability)', position: 'insideBottom', offset: -8 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="refusal"
            domain={[0, 1]}
            tickFormatter={pct}
            label={{ value: 'refusal', angle: -90, position: 'insideLeft' }}
          />
          <ZAxis range={[80, 80]} />

          {floor != null && (
            <ReferenceLine
              x={floor}
              stroke="#f59e0b"
              strokeDasharray="3 3"
              label={{ value: 'capability floor', position: 'top', fontSize: 11 }}
            />
          )}
          {baseline && (
            <ReferenceLine
              y={baseline.refuse_harmful}
              stroke="#94a3b8"
              strokeDasharray="2 2"
              label={{ value: 'baseline refusal', position: 'insideTopRight', fontSize: 11 }}
            />
          )}

          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ active, payload }: any) => {
              if (!active || !payload?.length) return null
              const p = payload[0].payload as FrontierRow
              return (
                <div className="rounded-md border bg-background px-3 py-2 text-xs shadow-sm">
                  <div className="font-medium">
                    rank {p.rank} · k {p.k.toFixed(2)}
                  </div>
                  <div>refusal {pct(p.refuse_harmful)}</div>
                  <div>factual {pct(p.factual_acc)}</div>
                  <div className="mt-1">
                    {p.degenerate
                      ? 'degenerate — output collapsed'
                      : p.accepted
                        ? 'holds the capability floor'
                        : 'below the capability floor'}
                  </div>
                </div>
              )
            }}
          />

          <Scatter name="rejected" data={rejected} fill="#94a3b8" fillOpacity={0.5} />
          <Scatter name="degenerate" data={degenerate} fill="#f59e0b" />
          <Scatter name="admissible" data={accepted} fill="#3b82f6" />
          <Scatter name="recommended" data={bestPoint} fill="#10b981" shape="star" />
        </ScatterChart>
      </ResponsiveContainer>

      <div className="mt-2 flex flex-wrap gap-4 text-xs text-muted-foreground">
        <Key color="#10b981" label="recommended" />
        <Key color="#3b82f6" label="holds the floor" />
        <Key color="#94a3b8" label="costs too much capability" />
        <Key color="#f59e0b" label="degenerate" />
      </div>

      {best ? (
        <div className="mt-4 rounded-lg border border-emerald-500/50 bg-emerald-50 p-4 text-sm dark:bg-emerald-950">
          <p className="font-medium">
            Recommended: rank {best.rank}, k {best.k.toFixed(2)}
          </p>
          <p className="text-muted-foreground">
            Refuses {pct(best.refuse_harmful)} of harmful prompts while holding factual
            accuracy at {pct(best.factual_acc)} — a drop of{' '}
            {(best.factual_drop * 100).toFixed(1)} points. Derive the subspace at exactly
            this rank so the basis matches, then write it with that strength.
          </p>
        </div>
      ) : (
        <div className="mt-4 rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 text-sm dark:bg-yellow-950">
          <p className="font-medium">No configuration cleared the capability floor</p>
          <p className="text-muted-foreground">
            On this model the remaining refusals look entangled with general capability:
            reaching them costs more factual accuracy than the floor allows. Raise the floor
            knowingly, or accept the single-direction edit.
          </p>
        </div>
      )}

      <FrontierTable frontier={frontier} best={best} />
    </div>
  )
}

function Key({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  )
}

function FrontierTable({ frontier, best }: { frontier: FrontierRow[]; best?: FrontierRow | null }) {
  return (
    <table className="mt-4 w-full text-sm">
      <thead>
        <tr className="border-b text-left text-xs text-muted-foreground">
          <th className="py-1 font-medium">rank</th>
          <th className="py-1 font-medium">k</th>
          <th className="py-1 font-medium">refusal</th>
          <th className="py-1 font-medium">factual</th>
          <th className="py-1 font-medium">verdict</th>
        </tr>
      </thead>
      <tbody>
        {frontier.map((r, i) => {
          const isBest = best && r.rank === best.rank && r.k === best.k
          return (
            <tr
              key={i}
              className={[
                'border-b last:border-0',
                r.degenerate ? 'bg-amber-50/60 dark:bg-amber-950/30' : '',
                isBest ? 'font-medium' : '',
              ].join(' ')}
            >
              <td className="py-1 font-mono">{r.rank}</td>
              <td className="py-1 font-mono">{r.k.toFixed(2)}</td>
              <td className="py-1 font-mono">
                {r.degenerate ? (
                  <span className="text-muted-foreground line-through">
                    {pct(r.refuse_harmful)}
                  </span>
                ) : (
                  pct(r.refuse_harmful)
                )}
              </td>
              <td className="py-1 font-mono">{pct(r.factual_acc)}</td>
              <td className="py-1 text-xs">
                {r.degenerate ? (
                  <span className="text-amber-700 dark:text-amber-300">
                    degenerate — this refusal number is meaningless
                  </span>
                ) : r.accepted ? (
                  <span className="text-emerald-700 dark:text-emerald-300">
                    holds the floor{isBest ? ' · recommended' : ''}
                  </span>
                ) : (
                  <span className="text-muted-foreground">costs too much capability</span>
                )}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
