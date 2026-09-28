import type { ReactNode } from 'react'

export const INPUT =
  'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50'

export const SMALL_BUTTON = 'rounded-md border px-2.5 py-1 text-xs font-medium hover:bg-muted disabled:opacity-40'

/** One step of the form: a numbered card with a title, what the step does, and an optional action slot. */
export function StepCard({ number, title, description, actions, children }: {
  number: number
  title: string
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="space-y-4 rounded-lg border bg-card p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">
            <span className="mr-2 text-muted-foreground">{number}</span>{title}
          </h2>
          {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
        </div>
        {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
      </div>
      {children}
    </section>
  )
}

/** A row of mutually exclusive choices, styled as a segmented control. */
export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T
  options: { value: T; label: string; title?: string }[]
  onChange: (value: T) => void
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap rounded-md border bg-muted/40 p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          title={option.title}
          onClick={() => onChange(option.value)}
          className={`rounded px-3 py-1 text-xs font-medium transition-colors ${
            value === option.value ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}
