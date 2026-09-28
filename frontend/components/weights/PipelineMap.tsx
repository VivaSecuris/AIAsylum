// The horizontal stepper for a task: its ordered steps with status and a
// "you are here" marker, so a newcomer sees the whole path and where they are
// instead of a flat grid of co-equal operations.
import { Check, Lock, ChevronRight } from 'lucide-react'

import type { WeightTaskStep } from '@/lib/api'

export type StepStatus = 'done' | 'current' | 'available' | 'locked'

export function PipelineMap({
  steps,
  statuses,
  activeIndex,
  onSelect,
}: {
  steps: WeightTaskStep[]
  statuses: StepStatus[]
  activeIndex: number
  onSelect: (index: number) => void
}) {
  return (
    <ol className="flex flex-wrap items-stretch gap-1">
      {steps.map((step, i) => {
        const status = statuses[i] ?? 'available'
        const isActive = i === activeIndex
        const locked = status === 'locked'
        return (
          <li key={`${step.kind}-${i}`} className="flex items-stretch">
            <button
              type="button"
              disabled={locked}
              onClick={() => onSelect(i)}
              aria-current={isActive ? 'step' : undefined}
              className={[
                'flex items-center gap-2 rounded-md border px-3 py-2 text-left text-sm transition',
                isActive ? 'border-primary bg-primary/10 ring-1 ring-primary' : 'border-border',
                locked ? 'cursor-not-allowed opacity-50' : 'hover:border-primary/60',
              ].join(' ')}
              title={step.reads}
            >
              <span
                className={[
                  'flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold',
                  status === 'done' ? 'bg-green-500 text-white'
                    : isActive ? 'bg-primary text-primary-foreground'
                    : locked ? 'bg-muted text-muted-foreground'
                    : 'border border-muted-foreground/40 text-muted-foreground',
                ].join(' ')}
              >
                {status === 'done' ? <Check className="h-3 w-3" /> : locked ? <Lock className="h-3 w-3" /> : i + 1}
              </span>
              <span className="whitespace-nowrap">{step.title}</span>
            </button>
            {i < steps.length - 1 && (
              <span className="flex items-center px-0.5 text-muted-foreground">
                <ChevronRight className="h-4 w-4" />
              </span>
            )}
          </li>
        )
      })}
    </ol>
  )
}
