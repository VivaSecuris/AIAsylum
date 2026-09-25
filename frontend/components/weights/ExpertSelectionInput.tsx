import { useMemo } from 'react'

// {"12": [3, 7], "15": "all"} -- the shape the API takes. Keys are strings
// because they travel as JSON object keys.
export type ExpertSelection = Record<string, number[] | 'all'>

export function parseExpertSelection(text: string): { selection: ExpertSelection; errors: string[] } {
  const selection: ExpertSelection = {}
  const errors: string[] = []
  text
    .split(/\n|;/)
    .map((line) => line.trim())
    .filter(Boolean)
    .forEach((line, i) => {
      const match = line.match(/^(\d+)\s*[:=]\s*(.+)$/)
      if (!match) {
        errors.push(`Line ${i + 1}: expected "layer: experts", got "${line}"`)
        return
      }
      const layer = String(Number(match[1]))
      const rhs = match[2].trim()
      if (rhs.toLowerCase() === 'all') {
        selection[layer] = 'all'
        return
      }
      const experts = rhs.split(/[,\s]+/).filter(Boolean).map(Number)
      if (!experts.length) {
        errors.push(`Line ${i + 1}: no experts listed for layer ${layer}`)
        return
      }
      if (experts.some((e) => !Number.isInteger(e) || e < 0)) {
        errors.push(`Line ${i + 1}: experts must be non-negative integers or "all"`)
        return
      }
      selection[layer] = Array.from(new Set(experts)).sort((a, b) => a - b)
    })
  return { selection, errors }
}

export function formatExpertSelection(selection: ExpertSelection | null | undefined): string {
  if (!selection) return ''
  return Object.entries(selection)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([layer, experts]) => `${layer}: ${experts === 'all' ? 'all' : experts.join(', ')}`)
    .join('\n')
}

export function ExpertSelectionInput({
  value,
  onChange,
  className,
}: {
  value: string
  onChange: (text: string) => void
  className?: string
}) {
  const { selection, errors } = useMemo(() => parseExpertSelection(value), [value])
  const layers = Object.keys(selection).length
  const named = Object.values(selection).reduce<number>((n, e) => n + (e === 'all' ? 0 : e.length), 0)
  const whole = Object.values(selection).filter((e) => e === 'all').length

  return (
    <div>
      <label className="text-xs font-medium" htmlFor="expert-selection">
        Experts to edit
      </label>
      <textarea
        id="expert-selection"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
        placeholder={'12: 3, 7\n15: all'}
        spellCheck={false}
        className={`${className ?? ''} font-mono`}
      />
      <p className="mt-1 text-xs text-muted-foreground">
        One layer per line as <code>layer: experts</code>. Expert indices are comma-separated;{' '}
        <code>all</code> takes every expert in that layer. Layers and experts are zero-based, as the
        routing table numbers them.
      </p>
      {errors.length > 0 ? (
        <ul className="mt-1 text-xs text-destructive">
          {errors.map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
      ) : value.trim() ? (
        <p className="mt-1 text-xs text-muted-foreground">
          {layers} layer{layers === 1 ? '' : 's'}, {named} named expert{named === 1 ? '' : 's'}
          {whole ? ` plus every expert in ${whole} layer${whole === 1 ? '' : 's'}` : ''}.
        </p>
      ) : null}
    </div>
  )
}
