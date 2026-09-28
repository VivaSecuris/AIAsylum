import type { ReactNode } from 'react'

import type { FormState, TestType } from '@/lib/create-test-config'
import { DEFAULT_MAX_TURNS, DEFAULT_NUM_MESSAGES, TEST_TYPES } from '@/lib/create-test-config'
import { BenchmarkSettings } from './BenchmarkSettings'
import { PromptPicker } from './PromptPicker'
import { INPUT, Segmented, StepCard } from './ui'

/** Step 1: what kind of test, and what it asks. */
export function TestDesignSection({ state, update, setTestType, variableNames, benchmarkSlot }: {
  state: FormState
  update: (patch: Partial<FormState>) => void
  setTestType: (type: TestType) => void
  variableNames: string[]
  /** Replaces the single-benchmark picker (suites choose several). */
  benchmarkSlot?: ReactNode
}) {
  const type = TEST_TYPES.find((t) => t.value === state.testType)
  return (
    <StepCard number={1} title="Test design" description={type?.description}>
      <div role="radiogroup" aria-label="Test type" className="flex flex-wrap gap-2">
        {TEST_TYPES.map((t) => (
          <button
            key={t.value}
            type="button"
            role="radio"
            aria-checked={state.testType === t.value}
            onClick={() => setTestType(t.value)}
            className={`rounded-md border px-4 py-2 text-sm font-medium transition-colors ${
              state.testType === t.value ? 'border-primary bg-primary text-primary-foreground' : 'hover:bg-muted'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {state.testType === 'one_shot' && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-sm font-medium">Test prompt</span>
            <Segmented label="Test prompt source" value={state.oneShotSource}
              options={[{ value: 'library', label: 'From library' }, { value: 'custom', label: 'Write my own' }]}
              onChange={(oneShotSource) => update({ oneShotSource })} />
          </div>
          {state.oneShotSource === 'library' ? (
            <PromptPicker mode="single" selectedIds={state.promptId ? [state.promptId] : []}
              onChange={(ids) => update({ promptId: ids[0] })} />
          ) : (
            <textarea aria-label="Custom test prompt" rows={4} value={state.customPrompt}
              placeholder="The prompt to send to the model under test. Use $name for a variable."
              onChange={(e) => update({ customPrompt: e.target.value })} className={INPUT} />
          )}
        </div>
      )}

      {state.testType === 'multi_shot' && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-sm font-medium">Test prompts</span>
            <Segmented label="Multi-shot prompt source" value={state.multiShotSource}
              options={[
                { value: 'library', label: 'From library' },
                { value: 'custom', label: 'Write my own' },
                { value: 'auto', label: 'Generated', title: 'Built-in sequential messages for context testing' },
              ]}
              onChange={(multiShotSource) => update({ multiShotSource })} />
          </div>
          {state.multiShotSource === 'library' && (
            <PromptPicker mode="multi" selectedIds={state.promptIds} onChange={(promptIds) => update({ promptIds })} />
          )}
          {state.multiShotSource === 'custom' && (
            <>
              <textarea aria-label="Custom test prompts, one per line" rows={6} value={state.customPrompts}
                placeholder="One prompt per line. Each line is sent in order in the same conversation."
                onChange={(e) => update({ customPrompts: e.target.value })} className={`${INPUT} font-mono`} />
              <p className="text-xs text-muted-foreground">Useful for context-window and needle-in-a-haystack tests.</p>
            </>
          )}
          {state.multiShotSource === 'auto' && (
            <label className="block max-w-xs text-xs font-medium">
              Number of messages
              <input type="number" min="1" max="100" value={state.numMessages} placeholder={String(DEFAULT_NUM_MESSAGES)}
                onChange={(e) => update({ numMessages: e.target.value })} className={INPUT} />
            </label>
          )}
        </div>
      )}

      {(state.testType === 'conversation' || state.testType === 'group_therapy') && (
        <label className="block max-w-xs text-sm font-medium">
          Max turns
          <input type="number" min="1" max="50" value={state.maxTurns} placeholder={String(DEFAULT_MAX_TURNS)}
            onChange={(e) => update({ maxTurns: e.target.value })} className={INPUT} />
          <span className="mt-1 block text-xs font-normal text-muted-foreground">
            How many rounds the doctor leads{state.testType === 'group_therapy' ? ' with the group' : ''}.
          </span>
        </label>
      )}

      {state.testType === 'benchmark' && (benchmarkSlot ?? <BenchmarkSettings state={state} update={update} />)}

      {variableNames.length > 0 && (
        <div className="space-y-2 rounded-md border bg-muted/30 p-3">
          <p className="text-sm font-medium">Prompt variables</p>
          <p className="text-xs text-muted-foreground">Values replace each $name in the prompt text before it is sent.</p>
          <div className="grid gap-3 sm:grid-cols-2">
            {variableNames.map((name) => (
              <label key={name} className="text-xs font-medium">
                ${name}
                <input value={state.variables[name] || ''} placeholder={`Value for ${name}`}
                  onChange={(e) => update({ variables: { ...state.variables, [name]: e.target.value } })}
                  className={INPUT} />
              </label>
            ))}
          </div>
        </div>
      )}
    </StepCard>
  )
}
