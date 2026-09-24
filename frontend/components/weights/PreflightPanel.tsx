import { AlertTriangle, Info, ShieldAlert } from 'lucide-react'

import type { PreflightCheck } from '@/lib/api'

interface Props {
  checks: PreflightCheck[]
  acknowledged: string[]
  onAcknowledge: (codes: string[]) => void
}

/**
 * Machine-state checks before a run.
 *
 * Severity is per-stage on the server: a resident Ollama model is advisory
 * before deriving a direction and blocking before writing weights, because the
 * consequence differs. Two codes are deliberately not acknowledgeable at all --
 * running out of disk part-way through a 6 GB write leaves a corrupt directory,
 * and no amount of clicking makes a missing torch import succeed.
 */
export function PreflightPanel({ checks, acknowledged, onAcknowledge }: Props) {
  if (!checks?.length) return null

  const blocking = checks.filter((c) => c.severity === 'blocking')
  const advisory = checks.filter((c) => c.severity === 'advisory')

  function toggle(code: string) {
    onAcknowledge(
      acknowledged.includes(code)
        ? acknowledged.filter((c) => c !== code)
        : [...acknowledged, code],
    )
  }

  return (
    <div className="space-y-3">
      {blocking.length > 0 && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
          <p className="mb-2 flex items-center gap-2 text-sm font-medium">
            <ShieldAlert className="h-4 w-4" />
            Blocking — this run will be refused
          </p>
          <ul className="space-y-2">
            {blocking.map((c) => (
              <li key={c.code} className="text-sm">
                {c.acknowledgeable ? (
                  <label className="flex cursor-pointer items-start gap-2">
                    <input
                      type="checkbox"
                      className="mt-1"
                      checked={acknowledged.includes(c.code)}
                      onChange={() => toggle(c.code)}
                    />
                    <span>
                      <span className="font-mono text-xs text-muted-foreground">{c.code}</span>
                      <br />
                      {c.message}
                    </span>
                  </label>
                ) : (
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
                    <span>
                      <span className="font-mono text-xs text-muted-foreground">{c.code}</span>
                      <br />
                      {c.message}
                      <em className="block text-xs text-muted-foreground">
                        This one cannot be overridden.
                      </em>
                    </span>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {advisory.length > 0 && (
        <div className="rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 dark:bg-yellow-950">
          <p className="mb-2 flex items-center gap-2 text-sm font-medium">
            <Info className="h-4 w-4" />
            Worth knowing before you start
          </p>
          <ul className="list-inside list-disc space-y-1 text-sm text-muted-foreground">
            {advisory.map((c) => (
              <li key={c.code}>{c.message}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
