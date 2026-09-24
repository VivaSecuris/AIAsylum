import { useState } from 'react'
import { useRouter } from 'next/router'
import { GitCompare } from 'lucide-react'

import { useCreateInterpRun } from '@/lib/hooks'
import { toast } from '@/lib/toast'

interface Props {
  sourceModel: string
  editedPath: string
  probePrompts?: string[]
}

/**
 * Launch the model_diff that shows what the edit actually changed.
 *
 * Entirely client-side: useCreateInterpRun already exists, so a backend
 * endpoint that proxied into the interp router would be pure duplication.
 *
 * The prompt defaults to one the direction was actually fitted against,
 * snapshotted by the direction run, rather than asking the user to invent one.
 */
export function CompareInInterpButton({ sourceModel, editedPath, probePrompts = [] }: Props) {
  const router = useRouter()
  const createRun = useCreateInterpRun()
  const [prompt, setPrompt] = useState(probePrompts[0] ?? '')
  const [open, setOpen] = useState(false)

  async function launch() {
    try {
      const run = await createRun.mutateAsync({
        mode: 'model_diff',
        model_a: sourceModel,
        model_b: editedPath,
        prompt_a: prompt,
        max_len: 512,
        device: 'auto',
        dtype: 'bfloat16',
      })
      toast.success('Started model comparison')
      router.push(`/interp/${run.id}`)
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Could not start the comparison')
    }
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-muted"
      >
        <GitCompare className="h-4 w-4" />
        Compare against baseline
      </button>
    )
  }

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <h3 className="text-sm font-semibold">Compare against the baseline</h3>
      <p className="mt-1 text-xs text-muted-foreground">
        Runs both models on one prompt and reports what diverged, layer by layer. Both are
        loaded, so this queues behind the same single model slot as everything else here.
      </p>

      <label className="mt-3 block text-xs text-muted-foreground">Prompt</label>
      {probePrompts.length > 0 && (
        <select
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={probePrompts.includes(prompt) ? prompt : ''}
          onChange={(e) => setPrompt(e.target.value)}
        >
          <option value="">Custom…</option>
          {probePrompts.map((p) => (
            <option key={p} value={p}>
              {p.length > 70 ? `${p.slice(0, 70)}…` : p}
            </option>
          ))}
        </select>
      )}
      <textarea
        rows={3}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="A prompt the direction was fitted against works best."
        className="mt-2 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
      />

      <div className="mt-3 flex gap-2">
        <button
          type="button"
          onClick={launch}
          disabled={!prompt.trim() || createRun.isPending}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {createRun.isPending ? 'Starting…' : 'Run comparison'}
        </button>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="rounded-md border px-4 py-2 text-sm hover:bg-muted"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
