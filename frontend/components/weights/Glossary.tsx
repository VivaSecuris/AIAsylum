// The help drawer: every jargon term with its definition, opened from the "?"
// button or by clicking an inline HelpTerm. Content comes from GET /glossary so
// the copy lives once in the backend beside the stage descriptions.
import { useEffect, useRef } from 'react'
import { X } from 'lucide-react'

import { useWeightGlossary } from '@/lib/hooks'

export function Glossary({
  open,
  onClose,
  focusTerm,
}: {
  open: boolean
  onClose: () => void
  focusTerm?: string | null
}) {
  const { data } = useWeightGlossary()
  const entries = data?.glossary ?? {}
  const focusRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (open && focusTerm && focusRef.current) {
      focusRef.current.scrollIntoView({ block: 'center' })
    }
  }, [open, focusTerm])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-label="Glossary">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <aside className="relative flex h-full w-full max-w-md flex-col border-l border-border bg-card shadow-xl">
        <div className="flex items-center justify-between border-b border-border p-4">
          <h2 className="text-lg font-semibold">Glossary</h2>
          <button type="button" onClick={onClose} aria-label="Close glossary" className="rounded p-1 hover:bg-muted">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="flex-1 space-y-4 overflow-auto p-4">
          {Object.entries(entries).map(([term, entry]) => {
            const isFocus = focusTerm === term
            return (
              <div
                key={term}
                ref={isFocus ? focusRef : undefined}
                className={`rounded-lg border p-3 ${isFocus ? 'border-primary bg-primary/5' : 'border-border'}`}
              >
                <p className="font-semibold">{term}</p>
                <p className="mt-1 text-sm text-muted-foreground">{entry.short}</p>
                {entry.long && <p className="mt-2 text-sm">{entry.long}</p>}
              </div>
            )
          })}
          {Object.keys(entries).length === 0 && (
            <p className="text-sm text-muted-foreground">Loading definitions…</p>
          )}
        </div>
      </aside>
    </div>
  )
}
