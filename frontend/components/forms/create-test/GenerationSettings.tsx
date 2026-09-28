import type { GenerationValues } from '@/lib/create-test-config'
import { DEFAULT_TEMPERATURE } from '@/lib/create-test-config'
import { INPUT } from './ui'

/** Per-step sampling settings, collapsed by default. Blank fields use the default. */
export function GenerationSettings({ value, onChange, showCot = true, showStrategies = false,
  defaultTemperature = DEFAULT_TEMPERATURE, defaultMaxTokens, minTokens = 1 }: {
  value: GenerationValues
  onChange: (value: GenerationValues) => void
  showCot?: boolean
  showStrategies?: boolean
  defaultTemperature?: number
  defaultMaxTokens?: number
  minTokens?: number
}) {
  const set = (patch: Partial<GenerationValues>) => onChange({ ...value, ...patch })
  const summary = [
    `temperature ${value.temperature.trim() || defaultTemperature}`,
    value.top_p.trim() ? `top-p ${value.top_p}` : null,
    value.max_tokens.trim() ? `max ${value.max_tokens} tokens` : null,
    showCot ? `ReACT ${value.enable_cot ? 'on' : 'off'}` : null,
    showStrategies ? `adaptive strategies ${value.use_dynamic_strategies ? 'on' : 'off'}` : null,
  ].filter(Boolean).join(' · ')

  return (
    <details className="rounded-md border bg-muted/30 px-4 py-2">
      <summary className="cursor-pointer text-sm font-medium">
        Advanced <span className="font-normal text-muted-foreground">· {summary}</span>
      </summary>
      <div className="grid gap-4 py-3 sm:grid-cols-3">
        <label className="text-xs font-medium">
          Temperature
          <input type="number" min="0" max="2" step="0.1" placeholder={String(defaultTemperature)}
            value={value.temperature} onChange={(e) => set({ temperature: e.target.value })} className={INPUT} />
          <span className="mt-1 block font-normal text-muted-foreground">0 is most repeatable; higher varies more.</span>
        </label>
        <label className="text-xs font-medium">
          Top-p
          <input type="number" min="0.01" max="1" step="0.05" placeholder="Model default"
            value={value.top_p} onChange={(e) => set({ top_p: e.target.value })} className={INPUT} />
          <span className="mt-1 block font-normal text-muted-foreground">Only sample from the most likely words.</span>
        </label>
        <label className="text-xs font-medium">
          Max tokens
          <input type="number" min={minTokens} max="32768" step="1" placeholder={defaultMaxTokens ? String(defaultMaxTokens) : "Model default"}
            value={value.max_tokens} onChange={(e) => set({ max_tokens: e.target.value })} className={INPUT} />
          <span className="mt-1 block font-normal text-muted-foreground">Longest reply per turn.</span>
        </label>
      </div>
      {(showCot || showStrategies) && (
        <div className="space-y-2 pb-3">
          {showCot && (
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" className="mt-1 rounded" checked={value.enable_cot}
                onChange={(e) => set({ enable_cot: e.target.checked })} />
              <span>
                Request ReACT reasoning
                <span className="block text-xs text-muted-foreground">Adds instructions to write reasoning text before the reply. This can change the behavior being tested. The transcript records generated text; it does not expose the model’s internal computation or execute actions.</span>
              </span>
            </label>
          )}
          {showStrategies && (
            <label className="flex items-start gap-2 text-sm">
              <input type="checkbox" className="mt-1 rounded" checked={value.use_dynamic_strategies}
                onChange={(e) => set({ use_dynamic_strategies: e.target.checked })} />
              <span>
                Adapt questioning strategy
                <span className="block text-xs text-muted-foreground">The doctor picks a probing strategy each turn from the conversation so far.</span>
              </span>
            </label>
          )}
        </div>
      )}
    </details>
  )
}
