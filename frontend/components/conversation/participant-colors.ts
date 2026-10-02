import type { GroupParticipant } from '@/lib/group-participants'

// Shared literals live under components so Tailwind includes every participant color.
const ACCENTS = [
  'border-slate-400 bg-slate-50 dark:bg-slate-900',
  'border-emerald-400 bg-emerald-50 dark:bg-emerald-950',
  'border-violet-400 bg-violet-50 dark:bg-violet-950',
  'border-amber-400 bg-amber-50 dark:bg-amber-950',
  'border-rose-400 bg-rose-50 dark:bg-rose-950',
  'border-cyan-400 bg-cyan-50 dark:bg-cyan-950',
]

export function participantAccent(participant: GroupParticipant): string {
  return ACCENTS[(participant.number - 1) % ACCENTS.length]
}
