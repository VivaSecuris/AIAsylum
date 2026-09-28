// A real "what's happening now" view, built from the accumulated SSE history
// rather than one overwriting line. The backend's Reporter emits "{name} ..."
// when a step starts and "{name} done in {t}" when it ends (plus per-item
// `count` ticks and training `metrics`), so we can render a phase checklist that
// fills in, a progress bar for the current phase, and the latest training loss.
import { Check, Loader2, AlertTriangle } from 'lucide-react'

import type { ProgressEvent } from '@/lib/hooks'

interface Phase {
  name: string
  status: 'running' | 'done' | 'failed'
  detail?: string
}

function derivePhases(history: ProgressEvent[]): { phases: Phase[]; notes: string[] } {
  const phases: Phase[] = []
  const notes: string[] = []
  const index = new Map<string, number>()

  const upsert = (name: string, status: Phase['status'], detail?: string) => {
    const at = index.get(name)
    if (at == null) {
      index.set(name, phases.length)
      phases.push({ name, status, detail })
    } else {
      phases[at] = { name, status, detail: detail ?? phases[at].detail }
    }
  }

  for (const ev of history) {
    const msg = (ev.message ?? '').trim()
    if (!msg) continue
    if (msg.endsWith('...')) {
      upsert(msg.replace(/\.\.\.$/, '').trim(), 'running')
    } else if (msg.includes(' done in ')) {
      const [name, took] = msg.split(' done in ')
      upsert(name.trim(), 'done', `${took}`)
    } else if (msg.includes(' FAILED after ')) {
      upsert(msg.split(' FAILED after ')[0].trim(), 'failed')
    } else {
      // A plain note (reporter.note) or a count tick — keep the most recent few.
      notes.push(msg)
    }
  }
  // Mark a still-"running" earlier phase as done once a later phase started.
  for (let i = 0; i < phases.length - 1; i++) {
    if (phases[i].status === 'running') phases[i].status = 'done'
  }
  return { phases, notes: notes.slice(-1) }
}

export function LiveStepView({
  history,
  progress,
  isConnected,
  terminal,
}: {
  history: ProgressEvent[]
  progress: ProgressEvent | null
  isConnected: boolean
  terminal?: 'completed' | 'failed' | 'cancelled' | null
}) {
  const { phases, notes } = derivePhases(history)
  const count = progress?.data as { done?: number; total?: number; percent?: number } | undefined
  const metrics = progress?.data as { step?: number; total?: number; loss?: number; eval_loss?: number } | undefined
  const showBar = !terminal && count?.total != null && count?.done != null
  const showLoss = !terminal && metrics?.loss != null

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2 text-sm">
        <span className={`h-2 w-2 rounded-full ${isConnected && !terminal ? 'animate-pulse bg-green-500' : 'bg-gray-400'}`} />
        <span className="font-medium">
          {terminal === 'completed' ? 'Complete'
            : terminal === 'failed' ? 'Failed'
            : terminal === 'cancelled' ? 'Cancelled'
            : isConnected ? 'Running' : 'Connecting…'}
        </span>
      </div>

      {phases.length > 0 && (
        <ol className="mt-3 space-y-1.5">
          {phases.map((p) => (
            <li key={p.name} className="flex items-center gap-2 text-sm">
              {p.status === 'done' ? (
                <Check className="h-4 w-4 shrink-0 text-green-500" />
              ) : p.status === 'failed' ? (
                <AlertTriangle className="h-4 w-4 shrink-0 text-red-500" />
              ) : (
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />
              )}
              <span className={p.status === 'done' ? 'text-muted-foreground' : ''}>{p.name}</span>
              {p.detail && <span className="text-xs text-muted-foreground">· {p.detail}</span>}
            </li>
          ))}
        </ol>
      )}

      {showBar && (
        <div className="mt-3">
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div className="h-full bg-primary transition-all" style={{ width: `${count!.percent ?? Math.round((count!.done! / count!.total!) * 100)}%` }} />
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {count!.done} / {count!.total}
          </p>
        </div>
      )}

      {showLoss && (
        <p className="mt-3 text-sm text-muted-foreground">
          step {metrics!.step}{metrics!.total ? ` / ${metrics!.total}` : ''} · loss {metrics!.loss!.toFixed(4)}
          {metrics!.eval_loss != null ? ` · eval ${metrics!.eval_loss.toFixed(4)}` : ''}
        </p>
      )}

      {!terminal && phases.length === 0 && (
        <p className="mt-3 text-sm text-muted-foreground">{progress?.message ?? 'Waiting to start…'}</p>
      )}
      {notes.length > 0 && !terminal && (
        <p className="mt-2 text-xs text-muted-foreground">{notes[0]}</p>
      )}
    </div>
  )
}
