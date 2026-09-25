import type { InterpAnalysis } from '@/lib/api'

interface Props {
  analyses: InterpAnalysis[]
  mode: string
  values: Record<string, boolean>
  onChange: (name: string, value: boolean) => void
}

/**
 * The capture and analysis flags, collapsed by default.
 *
 * All default off because each multiplies memory or search time, and the
 * causal ones are the expensive ones. Grouped by what they let you claim:
 * descriptive analyses rank components by how much they differ, while causal
 * ones intervene and re-measure. Presenting them as one undifferentiated list
 * would invite reading a contribution score as though it were evidence.
 */
export function AdvancedOptions({ analyses, mode, values, onChange }: Props) {
  // Only the analyses this mode actually computes. The backend rejects the
  // rest rather than capturing and discarding them, so offering them here
  // would only produce a rejected launch.
  const forMode = analyses.filter((a) => !a.modes || a.modes.includes(mode))
  const available = forMode.filter((a) => a.available !== false)
  const unavailable = forMode.filter((a) => a.available === false)
  const descriptive = available.filter((a) => a.claim === 'descriptive')
  const causal = available.filter((a) => a.claim === 'causal')

  if (!forMode.length) return null

  return (
    <details className="rounded-lg border p-4">
      <summary className="cursor-pointer text-sm font-medium">
        Advanced analyses{' '}
        <span className="font-normal text-muted-foreground">— expensive, all off by default</span>
      </summary>

      <div className="mt-4 space-y-5">
        <Group
          title="Captures and exploratory analyses"
          caption="Record activations and look for patterns. These measurements do not establish a causal mechanism."
          items={descriptive}
          values={values}
          onChange={onChange}
        />
        <Group
          title="Causal"
          caption="These intervene and run the model again to measure an effect on the selected prompt pair. They require additional forward passes."
          items={causal}
          values={values}
          onChange={onChange}
        />

        {unavailable.length > 0 && (
          <div>
            <p className="text-xs font-semibold">Not available</p>
            <p className="mb-2 text-xs text-muted-foreground">
              Implemented in outline but not yet producing a usable result. Listed so the
              gap is visible rather than silently missing.
            </p>
            <ul className="space-y-1">
              {unavailable.map((a) => (
                <li key={a.name} className="text-sm text-muted-foreground">
                  <span className="line-through">{a.label}</span>
                  <span className="block text-xs">{a.unavailable_reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </details>
  )
}

function Group({
  title,
  caption,
  items,
  values,
  onChange,
}: {
  title: string
  caption: string
  items: InterpAnalysis[]
  values: Record<string, boolean>
  onChange: (name: string, value: boolean) => void
}) {
  if (!items.length) return null
  return (
    <div>
      <p className="text-xs font-semibold">{title}</p>
      <p className="mb-2 text-xs text-muted-foreground">{caption}</p>
      <div className="space-y-2">
        {items.map((a) => (
          <label key={a.name} className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              className="mt-1"
              checked={!!values[a.name]}
              onChange={(e) => onChange(a.name, e.target.checked)}
            />
            <span>
              {a.label}
              <span className="block text-xs text-muted-foreground">{a.description}</span>
              <span className="block text-[11px] text-muted-foreground">{a.cost}</span>
            </span>
          </label>
        ))}
      </div>
    </div>
  )
}
