// A small coloured callout, extracted from the copies that had grown inside
// CompareTable, FrontierChart, SweepTable and ManifestCard. One tone vocabulary
// so every weights result speaks with the same visual grammar.
import type { ReactNode } from 'react'

export type BannerTone = 'good' | 'bad' | 'warn' | 'info' | 'neutral'

const TONES: Record<BannerTone, string> = {
  good: 'border-green-500/40 bg-green-500/10 text-green-900 dark:text-green-100',
  bad: 'border-red-500/40 bg-red-500/10 text-red-900 dark:text-red-100',
  warn: 'border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-100',
  info: 'border-blue-500/40 bg-blue-500/10 text-blue-900 dark:text-blue-100',
  neutral: 'border-border bg-muted/40 text-foreground',
}

export function Banner({
  tone = 'info',
  title,
  children,
  icon,
}: {
  tone?: BannerTone
  title?: ReactNode
  children?: ReactNode
  icon?: ReactNode
}) {
  return (
    <div className={`rounded-lg border p-3 text-sm ${TONES[tone]}`}>
      <div className="flex items-start gap-2">
        {icon && <span className="mt-0.5 shrink-0">{icon}</span>}
        <div className="min-w-0">
          {title && <p className="font-semibold">{title}</p>}
          {children && <div className={title ? 'mt-1' : ''}>{children}</div>}
        </div>
      </div>
    </div>
  )
}
