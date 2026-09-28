import type { TrialRow } from '@/lib/api'

interface Props {
  trials: TrialRow[]
  winner?: TrialRow | null
  maxRefusal?: number
}

const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`)

const REASONS: Record<string, string> = {
  ok: 'clears every gate',
  degenerate: 'degenerate — output collapsed',
  language_drift: 'answers in the wrong language',
  capability_cost: 'costs too much capability',
  sampled_degenerate: 'degenerate under sampling',
  sampled_language_drift: 'drifts into another language under sampling',
  sampled_capability_cost: 'loses capability under sampling',
}

/**
 * Every candidate the loop applied, in the order it tried them.
 *
 * The chart shows the trade; this shows the work. A row is one edit applied
 * to the real weights, scored, and restored. The sampled columns are filled
 * only for candidates that were re-applied for the serving-conditions pass,
 * so a winner that was chosen and then rejected there reads as such.
 */
export function TrialLog({ trials, winner, maxRefusal = 0.1 }: Props) {
  if (!trials?.length) return null
  return (
    <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
      <div className="border-b p-3">
        <h2 className="text-sm font-semibold">Trial log</h2>
        <p className="text-xs text-muted-foreground">
          One row per candidate, least destructive first. Each was applied to the real weights,
          scored greedily, and restored bit for bit before the next.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="px-3 py-2 font-medium">#</th>
              <th className="px-3 py-2 font-medium">rank</th>
              <th className="px-3 py-2 font-medium">k</th>
              <th className="px-3 py-2 font-medium">embeddings</th>
              <th className="px-3 py-2 font-medium">refusal</th>
              <th className="px-3 py-2 font-medium">factual</th>
              <th className="px-3 py-2 font-medium">drift</th>
              <th className="px-3 py-2 font-medium">sampled</th>
              <th className="px-3 py-2 font-medium">result</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((t) => {
              const isWinner = !!winner && winner.index === t.index
              const bad = !t.accepted || t.reason.startsWith('sampled_')
              return (
                <tr
                  key={t.index}
                  className={[
                    'border-b last:border-0',
                    isWinner ? 'bg-emerald-50/70 font-medium dark:bg-emerald-950/40' : '',
                    t.drifted || t.reason.endsWith('language_drift') ? 'bg-purple-50/60 dark:bg-purple-950/30' : '',
                    t.degenerate ? 'bg-amber-50/60 dark:bg-amber-950/30' : '',
                  ].join(' ')}
                >
                  <td className="px-3 py-2 font-mono text-xs">{t.index}</td>
                  <td className="px-3 py-2 font-mono">{t.rank}</td>
                  <td className="px-3 py-2 font-mono">{t.k.toFixed(2)}</td>
                  <td className="px-3 py-2">{t.include_embeddings ? 'edited' : 'untouched'}</td>
                  <td className="px-3 py-2 font-mono">
                    {bad ? <span className="text-muted-foreground line-through">{pct(t.refuse_harmful)}</span> : pct(t.refuse_harmful)}
                  </td>
                  <td className="px-3 py-2 font-mono">{pct(t.factual_acc)}</td>
                  <td className={`px-3 py-2 font-mono ${t.drifted ? 'text-purple-700 dark:text-purple-300' : ''}`}>
                    {pct(t.language_drift)}
                  </td>
                  <td className="px-3 py-2 font-mono text-xs">
                    {t.sampled
                      ? `${pct(t.sampled.refuse_harmful)} / ${pct(t.sampled.factual_acc)} / ${pct(t.sampled.language_drift)}`
                      : '—'}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {isWinner ? (
                      <span className="text-emerald-700 dark:text-emerald-300">
                        written{t.target_met ? ' · target met' : ` · refusal above ${pct(maxRefusal)} target`}
                      </span>
                    ) : (
                      <span className={bad ? 'text-muted-foreground' : 'text-emerald-700 dark:text-emerald-300'}>
                        {REASONS[t.reason] ?? t.reason}
                        {!bad && t.target_met ? ' · target met' : ''}
                      </span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="border-t px-3 py-2 text-[11px] text-muted-foreground">
        sampled = refusal / factual / drift under the provider&apos;s serving temperature and top-p, seeded.
        Drift is the share of answers written mostly outside the Latin script; a Chinese answer reads as
        0% refusal to the phrase matcher, which is why it is gated separately.
      </p>
    </div>
  )
}
