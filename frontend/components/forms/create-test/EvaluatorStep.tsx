import type { EvaluatorState } from '@/lib/create-test-config'
import { DEFAULT_EVALUATOR_TEMPERATURE } from '@/lib/create-test-config'
import { ModelSelector } from '../ModelSelector'
import { SystemPromptPicker } from './SystemPromptPicker'
import { StepCard } from './ui'
import { GenerationSettings } from './GenerationSettings'

/** Step 4: the model that scores the transcript once the test finishes. */
export function EvaluatorStep({ number, value, onChange, doctorLabel }: {
  number: number
  value: EvaluatorState
  onChange: (value: EvaluatorState) => void
  doctorLabel: string
}) {
  const set = (patch: Partial<EvaluatorState>) => onChange({ ...value, ...patch })
  return (
    <StepCard
      number={number}
      title="Evaluator"
      description="Scores the transcript on eight safety dimensions after the test finishes. You can also run this later from the run page."
      actions={
        <label className="flex items-center gap-2 text-sm font-medium">
          <input type="checkbox" className="rounded" checked={value.enabled} onChange={(e) => set({ enabled: e.target.checked })} />
          Analyse when the test completes
        </label>
      }
    >
      {value.enabled ? (
        <div className="space-y-4">
          <div>
            <ModelSelector
              label="Model"
              provider={value.provider}
              model={value.model}
              onProviderChange={(provider) => set({ provider, model: '' })}
              onModelChange={(model) => set({ model })}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Leave both blank to use the doctor ({doctorLabel || 'not chosen yet'}).{' '}
              {(value.provider || value.model) && (
                <button type="button" className="text-primary underline" onClick={() => set({ provider: '', model: '' })}>Use the doctor</button>
              )}
            </p>
          </div>
          <SystemPromptPicker role="evaluator" value={value.systemPrompt} onChange={(systemPrompt) => set({ systemPrompt })} />
          <div className="space-y-2">
            <span className="text-sm font-medium">Extra checks</span>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" className="rounded" checked={value.enable_cot_detection}
                onChange={(e) => set({ enable_cot_detection: e.target.checked })} />
              Chain-of-thought detection
              {value.enable_cot_detection && (
                <select aria-label="Chain-of-thought analysis depth" value={value.cot_analysis_mode}
                  onChange={(e) => set({ cot_analysis_mode: e.target.value as EvaluatorState['cot_analysis_mode'] })}
                  className="ml-2 rounded-md border bg-background px-2 py-1 text-xs">
                  <option value="full">Full</option>
                  <option value="partial">Partial</option>
                </select>
              )}
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" className="rounded" checked={value.enable_factuality_check}
                onChange={(e) => set({ enable_factuality_check: e.target.checked })} />
              Factuality and hallucination check
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" className="rounded" checked={value.enable_manipulation_analysis}
                onChange={(e) => set({ enable_manipulation_analysis: e.target.checked })} />
              Manipulation analysis
            </label>
          </div>
          <GenerationSettings
            value={{ temperature: value.temperature, top_p: value.top_p, max_tokens: value.max_tokens,
              enable_cot: value.enable_cot, use_dynamic_strategies: false }}
            defaultTemperature={DEFAULT_EVALUATOR_TEMPERATURE} minTokens={512}
            onChange={({ temperature, top_p, max_tokens, enable_cot }) => set({ temperature, top_p, max_tokens, enable_cot })}
          />
          <p className="text-xs text-muted-foreground">ReACT requests reasoning from the evaluator. Chain-of-thought detection separately analyzes reasoning in the recorded transcript.</p>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">Off. The run finishes without scores; you can analyse it later.</p>
      )}
    </StepCard>
  )
}
