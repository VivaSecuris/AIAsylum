import { AlertTriangle, CheckCircle2, HelpCircle } from 'lucide-react'

import type { SweepRow } from '@/lib/api'

interface Props {
  rows: SweepRow[]
  verdict?: string
  ablateDeltaPoints?: number | null
  factualDeltaPoints?: number | null
  layer?: number | null
}

/**
 * The causal check, rendered so its two traps are hard to fall into.
 *
 * First, the `ablate` row is the one to read and it is pinned to the top with
 * a caption saying why: ablation is scale-free, so unlike a large alpha it
 * cannot be confounded by a perturbation big enough to wreck the model.
 *
 * Second, a model that has collapsed into repetition emits no refusal phrases,
 * so a phrase-matching detector scores it 0% refusal -- indistinguishable from
 * ablation working perfectly. Those rows get their percentage struck through
 * and a worded badge rather than an icon, because a struck-out number is the
 * thing that stops someone lifting it into a slide.
 *
 * Deliberately not DataTable: that component hardcodes the row className with
 * no per-row hook, and the whole point here is per-row treatment.
 */
export function SweepTable({
  rows,
  verdict,
  ablateDeltaPoints,
  factualDeltaPoints,
  layer,
}: Props) {
  const hasCapability = rows.some((r) => r.factual_acc != null)
  if (!rows?.length) {
    return <p className="text-sm text-muted-foreground">This sweep recorded no rows.</p>
  }

  const ordered = [
    ...rows.filter((r) => r.label === 'ablate'),
    ...rows.filter((r) => r.label !== 'ablate'),
  ]

  return (
    <div className="space-y-4">
      <VerdictBanner
        verdict={verdict}
        deltaPoints={ablateDeltaPoints}
        factualDeltaPoints={factualDeltaPoints}
      />

      <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
        <div className="border-b p-3">
          <h2 className="text-sm font-semibold">Interventions</h2>
          <p className="text-xs text-muted-foreground">
            Ablation applies at every layer; addition is single-layer
            {layer != null ? ` (layer ${layer})` : ''}, with alpha as a multiple of that
            layer&apos;s measured residual norm.
          </p>
        </div>

        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="px-3 py-2 font-medium">intervention</th>
              <th className="px-3 py-2 font-medium">refusal</th>
              <th className="w-1/4 px-3 py-2 font-medium">rate</th>
              {hasCapability && <th className="px-3 py-2 font-medium">factual</th>}
              <th className="px-3 py-2 font-medium">n</th>
              <th className="px-3 py-2 font-medium">samples</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((row, i) => {
              const isAblate = row.label === 'ablate'
              const pct = row.refusal_rate * 100
              return (
                <tr
                  key={`${row.label}-${i}`}
                  className={[
                    'border-b align-top last:border-0',
                    row.degenerate
                      ? 'border-l-4 border-l-amber-500 bg-amber-50/60 dark:bg-amber-950/30'
                      : '',
                    isAblate && !row.degenerate ? 'border-l-4 border-l-emerald-500' : '',
                  ].join(' ')}
                >
                  <td className="px-3 py-2 font-mono text-xs">
                    {row.label}
                    {isAblate && (
                      <p className="mt-1 max-w-[16rem] font-sans text-[11px] leading-snug text-muted-foreground">
                        The row to trust — scale-free, and the inference-time preview of a
                        β=0 weight edit.
                      </p>
                    )}
                  </td>

                  <td className="px-3 py-2 font-mono">
                    {row.degenerate ? (
                      <span className="text-muted-foreground line-through">{pct.toFixed(1)}%</span>
                    ) : (
                      <span>{pct.toFixed(1)}%</span>
                    )}
                  </td>

                  <td className="px-3 py-2">
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className={`h-full rounded-full ${
                          row.degenerate ? 'bg-amber-400' : 'bg-blue-500'
                        }`}
                        style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
                      />
                    </div>
                    {row.degenerate && (
                      <p className="mt-1.5 inline-flex items-start gap-1 rounded bg-amber-100 px-1.5 py-1 text-[11px] leading-snug text-amber-900 dark:bg-amber-900/50 dark:text-amber-100">
                        <AlertTriangle className="mt-px h-3 w-3 shrink-0" />
                        <span>
                          Degenerate: output collapsed into repetition, so it emits no
                          refusal phrases. This rate is meaningless, not a success.
                        </span>
                      </p>
                    )}
                  </td>

                  {hasCapability && (
                    <td className="px-3 py-2 font-mono text-xs">
                      {row.factual_acc == null ? (
                        <span className="text-muted-foreground">—</span>
                      ) : (
                        <span
                          className={
                            row.factual_acc < 0.6 ? 'text-destructive' : undefined
                          }
                        >
                          {(row.factual_acc * 100).toFixed(0)}%
                        </span>
                      )}
                    </td>
                  )}
                  <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{row.n}</td>

                  <td className="px-3 py-2">
                    {row.samples?.length ? (
                      <details>
                        <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                          {row.samples.length} shown
                        </summary>
                        <div className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded border bg-muted/50 p-2 font-mono text-[11px]">
                          {row.samples.map((sample, j) => (
                            <div key={j} className={j ? 'mt-2 border-t pt-2' : ''}>
                              {sample}
                            </div>
                          ))}
                        </div>
                      </details>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>

        <p className="border-t px-3 py-2 text-[11px] text-muted-foreground">
          The engine keeps two sampled generations per intervention, so these are an
          illustration rather than the evidence — the refusal rate is measured over all n.
        </p>
      </div>
    </div>
  )
}

function VerdictBanner({
  verdict,
  deltaPoints,
  factualDeltaPoints,
}: {
  verdict?: string
  deltaPoints?: number | null
  factualDeltaPoints?: number | null
}) {
  const delta =
    deltaPoints == null ? null : `${deltaPoints > 0 ? '+' : ''}${deltaPoints.toFixed(1)} pts`
  const facDelta =
    factualDeltaPoints == null
      ? null
      : `${factualDeltaPoints > 0 ? '+' : ''}${factualDeltaPoints.toFixed(1)} pts`

  if (verdict === 'capability_cost') {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
        <div>
          <p className="font-medium">Lobotomised, not jailbroken ({delta})</p>
          <p className="text-muted-foreground">
            Ablation moved refusal, but factual accuracy moved {facDelta} with it. The
            direction is entangled with general capability here, so this is not evidence
            that refusal specifically was removed — and a weight edit would carry the same
            cost.
          </p>
        </div>
      </div>
    )
  }

  if (verdict === 'causal') {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-emerald-500/50 bg-emerald-50 p-4 text-sm dark:bg-emerald-950">
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
        <div>
          <p className="font-medium">Causal ({delta})</p>
          <p className="text-muted-foreground">
            Ablation moves refusal, so the direction is doing the work rather than merely
            correlating with it. A weight edit built on this should change behaviour.
            {facDelta
              ? ` The capability control held (${facDelta}), so this is about refusal rather than about the model getting worse.`
              : ' No capability control ran, so this says nothing about what the edit costs.'}
          </p>
        </div>
      </div>
    )
  }

  if (verdict === 'degenerate') {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-amber-500/50 bg-amber-50 p-4 text-sm dark:bg-amber-950">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <div>
          <p className="font-medium">Degenerate output</p>
          <p className="text-muted-foreground">
            The ablated model collapsed into repetition. Its refusal rate says nothing
            about the direction, and this sweep is not evidence either way.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-start gap-2 rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 text-sm dark:bg-yellow-950">
      <HelpCircle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-600" />
      <div>
        <p className="font-medium">Inconclusive{delta ? ` (${delta})` : ''}</p>
        <p className="text-muted-foreground">
          Ablation barely moves refusal. The direction does not look causal for this
          model, and weight surgery built on it would most likely fail silently rather
          than error.
        </p>
      </div>
    </div>
  )
}
