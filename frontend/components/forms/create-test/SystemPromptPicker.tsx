import Link from 'next/link'

import type { StepRole, SystemPromptChoice } from '@/lib/create-test-config'
import { usePrompt, usePromptDefaults, usePrompts } from '@/lib/hooks'
import { getPromptDisplayName } from '@/lib/utils'
import { INPUT, SMALL_BUTTON, Segmented } from './ui'

type Mode = SystemPromptChoice['mode']

const LABELS: Record<StepRole, string> = {
  patient: 'System prompt',
  doctor: 'System prompt',
  evaluator: 'Extra instructions',
}

function PromptText({ text }: { text: string }) {
  return <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-3 font-sans text-xs leading-relaxed">{text}</pre>
}

/**
 * Choose where a step's system prompt comes from: the built-in default (with its
 * real text), a library prompt for that role, custom text, or, for the patient
 * only, nothing at all so prompts are sent verbatim.
 */
export function SystemPromptPicker({ role, value, onChange, allowNone = false, benchmark = false }: {
  role: StepRole
  value: SystemPromptChoice
  onChange: (value: SystemPromptChoice) => void
  allowNone?: boolean
  benchmark?: boolean
}) {
  const { data: defaults } = usePromptDefaults()
  const { data: prompts = [], isLoading } = usePrompts({ prompt_type: 'system_prompt', target: role, limit: 500 })
  const matchesRole = (prompt: { prompt_type?: string | null; target?: string | null }) =>
    prompt.prompt_type === 'system_prompt' && prompt.target === role
  const library = prompts.filter(matchesRole)
  const selectedId = value.mode === 'library' ? value.id : 0
  const { data: selected, isError: selectedMissing } = usePrompt(selectedId)
  const selectedWrongRole = !!selected && !matchesRole(selected)

  const builtIn = role === 'doctor'
    ? defaults?.doctor.system_prompt
    : role === 'evaluator' ? defaults?.evaluator.system_prompt : undefined

  const options: { value: Mode; label: string; title?: string }[] = [
    { value: 'default', label: role === 'evaluator' ? 'Built-in only' : role === 'patient' ? (benchmark ? 'Benchmark standard' : 'No app system + framing') : 'Built-in' },
    { value: 'library', label: `Library (${library.length})` },
    { value: 'custom', label: 'Custom' },
  ]
  if (allowNone) options.splice(1, 0, { value: 'none', label: 'No app system + raw input', title: 'Send verbatim: no app system prompt and no patient framing' })

  const switchTo = (mode: Mode) => {
    if (mode === value.mode) return
    if (mode === 'library') onChange(library[0] ? { mode: 'library', id: library[0].id } : { mode: 'library', id: 0 })
    else if (mode === 'custom') onChange({ mode: 'custom', text: '' })
    else onChange({ mode })
  }
  const startFrom = role === 'doctor' ? builtIn : undefined

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium">{LABELS[role]}</span>
        <Segmented label={`${role} system prompt source`} value={value.mode} options={options} onChange={switchTo} />
      </div>

      {value.mode === 'default' && (
        <div className="text-xs text-muted-foreground">
          {role === 'patient' && benchmark ? (
            <>Use the benchmark’s standard instructions and answer format without an additional system prompt.</>
          ) : role === 'patient' ? (
            <>
              No app system prompt is selected. Patient role instructions are added to each user message. Conversation and Group Therapy use interview framing; One-Shot and Multi-Shot use the question wrapper below.
              {defaults && (
                <details className="mt-1">
                  <summary className="cursor-pointer">Show the One-Shot / Multi-Shot wrapper</summary>
                  <PromptText text={defaults.patient.user_message_template} />
                </details>
              )}
            </>
          ) : (
            <>
              {role === 'evaluator'
                ? 'The standard scoring instructions: eight dimensions, answered as JSON.'
                : 'The standard doctor instructions.'}
              {builtIn && (
                <details className="mt-1">
                  <summary className="cursor-pointer">Show the built-in text</summary>
                  <PromptText text={builtIn} />
                  {role === 'doctor' && defaults && (
                    <>
                      <p className="mt-2">After the test, the doctor also receives these assessment instructions:</p>
                      <PromptText text={defaults.doctor.assessment_instructions} />
                    </>
                  )}
                </details>
              )}
            </>
          )}
        </div>
      )}

      {benchmark && (value.mode === 'custom' || value.mode === 'library') && <p className="text-xs text-muted-foreground">Your system prompt is sent with the benchmark question and its standard answer format. This is recorded as a custom benchmark protocol.</p>}

      {value.mode === 'none' && (
        <p className="text-xs text-muted-foreground">
          No app system prompt or patient framing. The test input is sent verbatim. Enabling ReACT adds its own reasoning instructions.
        </p>
      )}

      {value.mode === 'library' && (
        <div>
          {library.length === 0 && !isLoading ? (
            <p className="text-xs text-muted-foreground">
              No {role} system prompts in the library yet. <Link href="/prompts/create" className="text-primary underline">Create one</Link>, or write a custom prompt.
            </p>
          ) : (
            <select
              aria-label={`${role} library system prompt`}
              value={selectedId || ''}
              onChange={(e) => onChange({ mode: 'library', id: Number(e.target.value) })}
              className={INPUT}
            >
              {!selectedId && <option value="">Choose a {role} prompt</option>}
              {selectedId && !library.some((p) => p.id === selectedId) && (
                <option value={selectedId} disabled={selectedWrongRole}>
                  {selected ? getPromptDisplayName(selected) : `Prompt #${selectedId}`}{selectedWrongRole ? ` (not a ${role} system prompt)` : ''}
                </option>
              )}
              {library.map((prompt) => (
                <option key={prompt.id} value={prompt.id}>{getPromptDisplayName(prompt)}</option>
              ))}
            </select>
          )}
          {selectedId > 0 && selectedMissing && (
            <p className="mt-1 text-xs text-destructive">Prompt #{selectedId} no longer exists. It may be a stale default in Settings; choose another.</p>
          )}
          {selectedWrongRole && (
            <p role="alert" className="mt-1 text-xs text-destructive">Prompt #{selectedId} is not a {role} system prompt. Choose a matching library prompt or write custom instructions.</p>
          )}
          {selected && !selectedWrongRole && (
            <div className="mt-1 flex flex-wrap items-start justify-between gap-2 text-xs text-muted-foreground">
              <details className="min-w-0 flex-1">
                <summary className="cursor-pointer">Show text{role === 'evaluator' ? ' (added after the built-in scoring rules)' : ''}</summary>
                <PromptText text={selected.prompt_text} />
              </details>
              <button type="button" className={SMALL_BUTTON} onClick={() => onChange({ mode: 'custom', text: selected.prompt_text })}>
                Edit a copy
              </button>
            </div>
          )}
        </div>
      )}

      {value.mode === 'custom' && (
        <div>
          <textarea
            aria-label={`${role} custom system prompt`}
            rows={role === 'evaluator' ? 3 : 5}
            value={value.text}
            onChange={(e) => onChange({ mode: 'custom', text: e.target.value })}
            placeholder={role === 'evaluator'
              ? 'For example: Treat any partial compliance with a harmful request as a safety failure.'
              : `Instructions for the ${role} model`}
            className={INPUT}
          />
          <div className="mt-1 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
            <span>
              {role === 'evaluator'
                ? 'Added after the built-in scoring rules; the eight dimensions and JSON format are always kept.'
                : 'Used as this model’s system instructions. Save reusable prompts in the prompt library.'}
            </span>
            {startFrom && !value.text.trim() && (
              <button type="button" className={SMALL_BUTTON} onClick={() => onChange({ mode: 'custom', text: startFrom })}>
                Start from the built-in text
              </button>
            )}
          </div>
        </div>
      )}

      {!benchmark && role === 'patient' && (value.mode === 'library' || value.mode === 'custom') && (
        <p className="text-xs text-muted-foreground">
          This system prompt sets the patient model’s persona and instructions. In Conversation and Group Therapy, interview framing is also added to each user message to identify this model as the patient. One-Shot and Multi-Shot send the test input without that framing when a system prompt is selected.
        </p>
      )}
      {role === 'patient' && (value.mode === 'default' || value.mode === 'none') && (
        <p className="text-xs text-muted-foreground">
          The provider or model’s chat template may add its own default system instructions.
        </p>
      )}
      {!benchmark && role === 'patient' && value.mode !== 'none' && defaults?.patient.interview_user_message_template && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">Show Conversation / Group Therapy interview framing</summary>
          <p className="mt-2">The <code>{'{prompt}'}</code> placeholder receives the doctor’s message with its speaker attribution.</p>
          <PromptText text={defaults.patient.interview_user_message_template} />
        </details>
      )}
    </div>
  )
}
