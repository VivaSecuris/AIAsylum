import type { StepState } from '@/lib/create-test-config'
import { ModelSelector } from '../ModelSelector'
import { GenerationSettings } from './GenerationSettings'
import { SystemPromptPicker } from './SystemPromptPicker'

/** The settings for one LLM in the run: its model, its system prompt and its sampling. */
export function RoleFields({ role, step, onChange, modelLabel = 'Model', allowNone = false,
  showSystemPrompt = true, showGeneration = true, showCot = true, showStrategies = false, defaultTemperature, defaultMaxTokens, benchmark = false, doctorGoal }: {
  role: 'patient' | 'doctor'
  step: StepState
  onChange: (step: StepState) => void
  modelLabel?: string
  allowNone?: boolean
  showSystemPrompt?: boolean
  showGeneration?: boolean
  showCot?: boolean
  showStrategies?: boolean
  defaultTemperature?: number
  defaultMaxTokens?: number
  benchmark?: boolean
  doctorGoal?: { value: string; onChange: (value: string) => void }
}) {
  return (
    <div className="space-y-4">
      <ModelSelector
        label={modelLabel}
        provider={step.provider}
        model={step.model}
        onProviderChange={(provider) => onChange({ ...step, provider, model: '' })}
        onModelChange={(model) => onChange({ ...step, model })}
      />
      {showSystemPrompt && (
        <SystemPromptPicker benchmark={benchmark} role={role} value={step.systemPrompt} allowNone={allowNone}
          onChange={(systemPrompt) => onChange({ ...step, systemPrompt })} />
      )}
      {role === 'doctor' && doctorGoal && <DoctorGoalField value={doctorGoal.value} onChange={doctorGoal.onChange} />}
      {showGeneration && (
        <GenerationSettings defaultTemperature={defaultTemperature} defaultMaxTokens={defaultMaxTokens} value={step.generation} showCot={showCot} showStrategies={showStrategies}
          showContextWindow={step.provider === 'ollama'}
          onChange={(generation) => onChange({ ...step, generation })} />
      )}
    </div>
  )
}

export function DoctorGoalField({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return <div className="space-y-1">
    <label htmlFor="doctor-conversation-goal" className="text-sm font-medium">Doctor conversation goal (optional)</label>
    <textarea id="doctor-conversation-goal" value={value} onChange={(event) => onChange(event.target.value)} maxLength={8000} rows={3}
      aria-describedby="doctor-conversation-goal-help" placeholder="What should the doctor explore or assess during this conversation?"
      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" />
    <p id="doctor-conversation-goal-help" className="text-xs text-muted-foreground">Guides the doctor’s questions and final assessment. Sent only to the doctor as additional user instructions; the selected system prompts are preserved.</p>
  </div>
}
