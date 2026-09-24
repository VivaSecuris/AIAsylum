import type { DirectionOption } from '@/lib/api'

interface Props {
  directions: DirectionOption[]
  value: number | null
  onChange: (id: number | null) => void
  minUsableAuc?: number
}

/**
 * Pick the direction a sweep or surgery will consume.
 *
 * Unusable directions are shown rather than hidden, with the AUC visible and a
 * chip saying why. Hiding them would leave the user wondering where their run
 * went; showing them greyed makes choosing one a deliberate act, which the
 * launcher then turns into an explicit acknowledgement.
 *
 * `artifacts_present` is checked server-side against the filesystem: a row
 * outlives its files whenever a run is deleted or runs/ is cleaned, and those
 * are unselectable because nothing can be done with them.
 */
export function DirectionPicker({ directions, value, onChange, minUsableAuc = 0.9 }: Props) {
  if (!directions.length) {
    return (
      <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
        No directions yet. Derive one first — a sweep and a surgery both consume one.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {directions.map((d) => {
        const disabled = !d.artifacts_present
        const unusable = d.usable === false
        const selected = value === d.id

        return (
          <label
            key={d.id}
            className={[
              'flex cursor-pointer items-start gap-3 rounded-lg border p-3 text-sm transition-colors',
              selected ? 'border-primary bg-primary/5' : 'hover:bg-muted/50',
              disabled ? 'cursor-not-allowed opacity-50' : '',
              unusable && !disabled ? 'opacity-70' : '',
            ].join(' ')}
          >
            <input
              type="radio"
              name="direction"
              className="mt-1"
              disabled={disabled}
              checked={selected}
              onChange={() => onChange(d.id)}
            />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">#{d.id}</span>
                <span className="truncate font-mono text-xs text-muted-foreground">
                  {d.source_model}
                </span>
                {d.objective && (
                  <span className="rounded bg-muted px-1.5 py-0.5 text-[10px]">{d.objective}</span>
                )}
              </div>

              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span className="font-mono">layer {d.layer ?? '—'}</span>
                <span className="font-mono">AUC {d.auc != null ? d.auc.toFixed(3) : '—'}</span>
                <span className="font-mono">d {d.cohens_d != null ? d.cohens_d.toFixed(2) : '—'}</span>

                {unusable && (
                  <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] text-amber-900 dark:bg-amber-900/50 dark:text-amber-100">
                    below {minUsableAuc.toFixed(2)} — does not separate
                  </span>
                )}
                {d.has_sweep ? (
                  <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200">
                    causality checked
                  </span>
                ) : (
                  <span className="rounded bg-muted px-1.5 py-0.5 text-[10px]">no sweep yet</span>
                )}
                {disabled && (
                  <span className="rounded bg-destructive/15 px-1.5 py-0.5 text-[10px] text-destructive">
                    artifacts deleted
                  </span>
                )}
              </div>
            </div>
          </label>
        )
      })}
    </div>
  )
}
