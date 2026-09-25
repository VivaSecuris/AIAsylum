export interface PatchOptions {
  component: 'layer' | 'head' | 'neuron'
  layers: string
  positions: string
  units: string
}

const INPUT = 'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm'

export function parsePatchOptions(options: PatchOptions) {
  const errors: string[] = []
  function indices(text: string, label: string): number[] | undefined {
    if (!text.trim()) return undefined
    if (!/^\d+(\s*,\s*\d+)*$/.test(text.trim())) {
      errors.push(`${label}: enter nonnegative integers separated by commas.`)
      return undefined
    }
    const values = text.split(',').map(Number)
    if (values.length > 64 || new Set(values).size !== values.length) errors.push(`${label}: use at most 64 distinct indices.`)
    return values
  }
  const layers = indices(options.layers, 'Patch layers')
  const positions = indices(options.positions, 'Patch positions')
  let units: [number, number][] | undefined
  if (options.component !== 'layer' && options.units.trim()) {
    if (!/^\d+:\d+(\s*,\s*\d+:\d+)*$/.test(options.units.trim())) errors.push('Use block:unit pairs, such as 0:1, 2:3.')
    else {
      units = options.units.split(',').map((entry) => entry.trim().split(':').map(Number) as [number, number])
      if (units.length > 64 || new Set(units.map(String)).size !== units.length) errors.push('Use at most 64 distinct block:unit pairs.')
      if (layers && (layers.some((block) => !units!.some(([layer]) => layer === block)) || units.some(([block]) => !layers.includes(block)))) errors.push('Patch layers must match the blocks in the selected units, or leave layers blank.')
    }
  }
  return { errors, request: {
    patch_components: options.component,
    patch_layers: layers,
    patch_positions: positions,
    patch_heads: options.component === 'head' ? units : undefined,
    patch_neurons: options.component === 'neuron' ? units : undefined,
  } }
}

export function PatchingOptions({ value, onChange }: { value: PatchOptions; onChange: (value: PatchOptions) => void }) {
  return <div className="space-y-3 rounded-lg border p-4">
    <h3 className="text-sm font-medium">Choose the intervention</h3>
    <p className="text-xs text-muted-foreground">Each selected component and position is tested separately in both directions. Larger selections require more model passes. Indices start at zero; invalid indices fail explicitly.</p>
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="text-xs">Component<select aria-label="Patch component" className={INPUT} value={value.component} onChange={(event) => onChange({ ...value, component: event.target.value as PatchOptions['component'], units: '' })}>
        <option value="layer">Residual state</option><option value="head">Attention head</option><option value="neuron">MLP neuron</option>
      </select></label>
      <label className="text-xs">{value.component === 'layer' ? 'Hidden-state indices' : 'Transformer blocks'}<input aria-label="Patch layers" className={INPUT} value={value.layers} onChange={(event) => onChange({ ...value, layers: event.target.value })} placeholder={value.component === 'layer' ? 'Blank: all states; example: 0, 4, 8' : 'Blank: largest-divergence block'} /></label>
      <label className="text-xs">Aligned token positions<input aria-label="Patch positions" className={INPUT} value={value.positions} onChange={(event) => onChange({ ...value, positions: event.target.value })} placeholder="Blank: last aligned token; example: 0, 2" /></label>
      {value.component !== 'layer' && <label className="text-xs">Specific {value.component === 'head' ? 'heads' : 'neurons'} (block:unit)<input aria-label="Patch units" className={INPUT} value={value.units} onChange={(event) => onChange({ ...value, units: event.target.value })} placeholder="Example: 0:1, 2:3" /></label>}
    </div>
    <p className="text-xs text-muted-foreground">{value.component === 'head' ? 'Head patching automatically captures attention and Q/K/V. Blank units test all heads in each selected block.' : value.component === 'neuron' ? 'Neuron patching automatically captures internal MLP activations. Blank units select the 50 largest contribution differences per selected block.' : 'State 0 is the embedding output; the final index is the final hidden state. Intermediate indices are inputs to their transformer block.'}</p>
  </div>
}
