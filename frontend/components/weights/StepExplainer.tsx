// Explains the current pipeline step before the form: what it does, when to use
// it, what you'll get, and how to read the result. Copy comes from the task
// registry (step) with the stage's own description as a fallback.
import type { WeightTaskStep, WeightStage } from '@/lib/api'

export function StepExplainer({ step, stage }: { step: WeightTaskStep; stage?: WeightStage }) {
  const rows: Array<[string, string | undefined]> = [
    ['What it does', step.does ?? stage?.description],
    ['When to use it', step.when],
    ["What you'll get", step.you_get ?? (stage ? `Writes ${stage.writes}.` : undefined)],
    ['How to read it', step.reads],
  ]
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-4">
      <h3 className="text-sm font-semibold">{step.title}</h3>
      <dl className="mt-2 space-y-1.5 text-sm">
        {rows
          .filter(([, v]) => !!v)
          .map(([label, value]) => (
            <div key={label} className="grid grid-cols-[7.5rem_1fr] gap-2">
              <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</dt>
              <dd className="text-muted-foreground">{value}</dd>
            </div>
          ))}
      </dl>
    </div>
  )
}
