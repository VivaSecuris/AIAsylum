// A lightweight tooltip and an inline "help term" chip. No dependency: a
// CSS/absolute-positioned popover shown on hover and keyboard focus, so a reader
// can hover any jargon word and get its one-line definition. The full glossary
// lives in the Glossary drawer; this is the first line of help.
import { useState, type ReactNode } from 'react'

export function Tooltip({ content, children }: { content: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open && (
        <span
          role="tooltip"
          className="absolute bottom-full left-1/2 z-50 mb-1 w-64 -translate-x-1/2 rounded-md border border-border bg-popover p-2 text-left text-xs font-normal leading-snug text-popover-foreground shadow-lg"
        >
          {content}
        </span>
      )}
    </span>
  )
}

// Wraps a jargon term. `def` is the tooltip text; `onOpenGlossary` (optional)
// deep-links into the drawer for the long form.
export function HelpTerm({
  term,
  def,
  onOpenGlossary,
}: {
  term: ReactNode
  def: ReactNode
  onOpenGlossary?: () => void
}) {
  return (
    <Tooltip
      content={
        <span>
          {def}
          {onOpenGlossary && (
            <button
              type="button"
              onClick={onOpenGlossary}
              className="ml-1 font-medium text-primary hover:underline"
            >
              more
            </button>
          )}
        </span>
      }
    >
      <button
        type="button"
        onClick={onOpenGlossary}
        className="cursor-help border-b border-dotted border-muted-foreground/60 text-inherit"
        aria-label={typeof term === 'string' ? `Definition of ${term}` : 'definition'}
      >
        {term}
      </button>
    </Tooltip>
  )
}
