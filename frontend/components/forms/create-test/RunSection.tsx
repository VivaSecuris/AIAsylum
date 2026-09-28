import type { FormState } from '@/lib/create-test-config'
import { seedTargets, stepsForType } from '@/lib/create-test-config'
import { INPUT, StepCard } from './ui'

/** The last step: name, seed, what will run, and anything that blocks it. */
export function RunSection({ number, state, update, errors, warnings, summary, attempted,
  namePlaceholder = 'Shown in the run list instead of the ID' }: {
  number: number
  state: FormState
  update: (patch: Partial<FormState>) => void
  errors: string[]
  warnings: string[]
  summary: string
  /** After a Create click the remaining items are shown as errors, before it as a checklist. */
  attempted: boolean
  namePlaceholder?: string
}) {
  const seeds = seedTargets(state)
  const showSeed = stepsForType(state.testType).seed
  return (
    <StepCard number={number} title="Run">
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="text-sm font-medium">
          Name <span className="font-normal text-muted-foreground">(optional)</span>
          <input value={state.name} maxLength={200} placeholder={namePlaceholder}
            onChange={(e) => update({ name: e.target.value })} className={INPUT} />
        </label>
        {showSeed && (
          <label className="text-sm font-medium">
            Seed <span className="font-normal text-muted-foreground">(optional)</span>
            <input type="number" min="0" max="4294967295" step="1" value={state.seed} placeholder="Random"
              onChange={(e) => update({ seed: e.target.value })} className={INPUT} />
            <span className="mt-1 block text-xs font-normal text-muted-foreground">
              {state.seed.trim() === ''
                ? 'Set a seed to make sampling repeatable where the provider supports it.'
                : [
                    seeds.applied.length ? `Applied to: ${seeds.applied.join(', ')}.` : '',
                    seeds.skipped.length ? `Not applied to ${seeds.skipped.join(', ')}: that provider does not accept a seed.` : '',
                  ].filter(Boolean).join(' ')}
            </span>
          </label>
        )}
      </div>
      <p className="rounded-md bg-muted/40 p-3 text-sm">{summary}</p>
      {errors.length > 0 && (
        <div role={attempted ? 'alert' : undefined}
          className={`rounded-md border p-3 text-sm ${attempted ? 'border-destructive/40 bg-destructive/5' : 'bg-muted/20'}`}>
          <p className={`font-medium ${attempted ? 'text-destructive' : ''}`}>{attempted ? 'Before this can run:' : 'To start:'}</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {errors.map((error) => <li key={error}>{error}</li>)}
          </ul>
        </div>
      )}
      {warnings.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-5 text-xs text-muted-foreground">
          {warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}
    </StepCard>
  )
}
