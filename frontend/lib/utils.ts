import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
import { format, formatDistanceToNow } from 'date-fns'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(date: string | Date | null | undefined): string {
  if (!date) return ''
  const d = typeof date === 'string' ? new Date(date) : date
  return format(d, 'PPp')
}

export function formatDateTime(date: string | Date | null | undefined): string {
  if (!date) return ''
  const d = typeof date === 'string' ? new Date(date) : date
  // Format: "Jan 26, 2025 at 3:45 PM" - clear date and time
  return format(d, 'MMM d, yyyy \'at\' h:mm a')
}

export function formatRelativeDate(date: string | Date | null | undefined): string {
  if (!date) return ''
  const d = typeof date === 'string' ? new Date(date) : date
  return formatDistanceToNow(d, { addSuffix: true })
}

/** Opaque name pattern from import script: jailbreak_<index>_<hash> */
const OPAQUE_PROMPT_NAME = /^jailbreak_\d+_[a-f0-9]{8}$/i

function formatTechnique(technique: string): string {
  return technique
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

/**
 * Returns a user-friendly display name for a prompt so users know what they selected.
 * Prefers technique/source over opaque names like "jailbreak_1441_032d5ff3".
 */
export function getPromptDisplayName(prompt: {
  name: string
  category?: string
  metadata?: Record<string, unknown>
  prompt_text?: string
}): string {
  const technique = prompt.metadata?.jailbreak_technique as string | undefined
  const source = prompt.metadata?.source as string | undefined
  const previewLen = 45

  if (prompt.category === 'adversarial' && technique) {
    const techniqueLabel = formatTechnique(technique)
    const preview =
      prompt.prompt_text?.trim().replace(/\s+/g, ' ').slice(0, previewLen)
    if (preview && prompt.name.match(OPAQUE_PROMPT_NAME))
      return `${techniqueLabel} — ${preview}${(prompt.prompt_text?.length ?? 0) > previewLen ? '…' : ''}`
    if (source && source !== 'unknown') return `${techniqueLabel} (${source})`
    return techniqueLabel
  }

  if (prompt.name.match(OPAQUE_PROMPT_NAME)) {
    if (technique) return formatTechnique(technique)
    const preview =
      prompt.prompt_text?.trim().replace(/\s+/g, ' ').slice(0, previewLen)
    if (preview) return `Jailbreak — ${preview}${(prompt.prompt_text?.length ?? 0) > previewLen ? '…' : ''}`
    return 'Jailbreak prompt'
  }

  return prompt.name
}
