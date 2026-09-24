import { useState } from 'react'
import { useRouter } from 'next/router'
import { Brain, Play, Trash2 } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { AdvancedOptions } from '@/components/interp/AdvancedOptions'
import {
  useInterpModes,
  useInterpRuns,
  useCreateInterpRun,
  useDeleteInterpRun,
} from '@/lib/hooks'
import type { InterpMode } from '@/lib/api'
import { toast } from '@/lib/toast'
import { formatDateTime } from '@/lib/utils'

const INPUT =
  'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

export default function InterpIndexPage() {
  const router = useRouter()
  const { data: modeInfo, isLoading: loadingModes } = useInterpModes()
  const { data: runs = [], isLoading: loadingRuns } = useInterpRuns({ limit: 50 })
  const createRun = useCreateInterpRun()
  const deleteRun = useDeleteInterpRun()

  const [mode, setMode] = useState<InterpMode>('single')
  const [modelA, setModelA] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [modelB, setModelB] = useState('')
  const [promptA, setPromptA] = useState('')
  const [promptB, setPromptB] = useState('')
  const [promptList, setPromptList] = useState('')
  const [maxLen, setMaxLen] = useState(512)
  const [attention, setAttention] = useState(false)
  // The capture and causal flags, all off by default. Previously these were
  // hardcoded off in the backend, so the minimal-circuit search could not be
  // reached from the app at all.
  const [advanced, setAdvanced] = useState<Record<string, boolean>>({})

  const spec = modeInfo?.modes.find((m) => m.name === mode)

  function selectMode(next: InterpMode) {
    setMode(next)
    const allowed = new Set(
      (modeInfo?.analyses ?? [])
        .filter((a) => !a.modes || a.modes.includes(next))
        .filter((a) => a.available !== false)
        .map((a) => a.name),
    )
    setAdvanced((prev) =>
      Object.fromEntries(Object.entries(prev).filter(([k, v]) => v && allowed.has(k))),
    )
  }
  const needs = (field: string) => spec?.needs.includes(field) ?? false
  const ceiling = modeInfo?.limits.max_len_ceiling ?? 1024

  async function launch() {
    try {
      const run = await createRun.mutateAsync({
        mode,
        model_a: modelA.trim(),
        model_b: needs('model_b') ? modelB.trim() : undefined,
        prompt_a: needs('prompt_a') ? promptA : undefined,
        prompt_b: needs('prompt_b') ? promptB : undefined,
        prompts: needs('prompts')
          ? promptList.split('\n').map((p) => p.trim()).filter(Boolean)
          : undefined,
        max_len: maxLen,
        enable_attention_capture: attention,
        device: 'auto',
        dtype: 'bfloat16',
        ...advanced,
      })
      toast.success(`Started ${mode} analysis`)
      router.push(`/interp/${run.id}`)
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Could not start the analysis')
    }
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Brain className="h-7 w-7" />
          <div>
            <h1 className="text-3xl font-bold">Interpretability</h1>
            <p className="text-sm text-muted-foreground">
              Look inside a model: what activates, where two prompts diverge, and what a
              weight edit actually changed.
            </p>
          </div>
        </div>

        {/* Memory warnings are the difference between a slow run and an apparent hang. */}
        {(modeInfo?.warnings?.length ?? 0) > 0 && (
          <div className="rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 dark:bg-yellow-950">
            <p className="mb-2 text-sm font-medium">Before running an analysis</p>
            <ul className="list-inside list-disc space-y-1 text-sm text-muted-foreground">
              {modeInfo!.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h2 className="mb-4 text-lg font-semibold">New analysis</h2>

          {loadingModes ? (
            <LoadingSpinner />
          ) : (
            <div className="space-y-4">
              <div>
                <label className="text-xs text-muted-foreground">Mode</label>
                <select
                  value={mode}
                  onChange={(e) => selectMode(e.target.value as InterpMode)}
                  className={INPUT}
                >
                  {modeInfo?.modes.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.label}
                    </option>
                  ))}
                </select>
                {spec && (
                  <p className="mt-1 text-xs text-muted-foreground">{spec.description}</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-muted-foreground">
                    {needs('model_b') ? 'Model A (baseline)' : 'Model'}
                  </label>
                  <input
                    value={modelA}
                    onChange={(e) => setModelA(e.target.value)}
                    placeholder="Qwen/Qwen2.5-0.5B-Instruct"
                    className={INPUT}
                  />
                </div>
                {needs('model_b') && (
                  <div>
                    <label className="text-xs text-muted-foreground">Model B (modified)</label>
                    <input
                      value={modelB}
                      onChange={(e) => setModelB(e.target.value)}
                      placeholder="models/ablated"
                      className={INPUT}
                    />
                    <p className="mt-1 text-xs text-muted-foreground">
                      A directory written by <code>aiasylum weights ablate</code>.
                    </p>
                  </div>
                )}
              </div>

              {needs('prompt_a') && (
                <div>
                  <label className="text-xs text-muted-foreground">
                    {needs('prompt_b') ? 'Prompt A' : 'Prompt'}
                  </label>
                  <textarea
                    value={promptA}
                    onChange={(e) => setPromptA(e.target.value)}
                    rows={2}
                    className={INPUT}
                  />
                </div>
              )}

              {needs('prompt_b') && (
                <div>
                  <label className="text-xs text-muted-foreground">Prompt B</label>
                  <textarea
                    value={promptB}
                    onChange={(e) => setPromptB(e.target.value)}
                    rows={2}
                    className={INPUT}
                  />
                </div>
              )}

              {needs('prompts') && (
                <div>
                  <label className="text-xs text-muted-foreground">
                    Prompts, one per line (zero-shot first)
                  </label>
                  <textarea
                    value={promptList}
                    onChange={(e) => setPromptList(e.target.value)}
                    rows={4}
                    className={INPUT}
                  />
                </div>
              )}

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-muted-foreground">
                    Max sequence length (ceiling {ceiling})
                  </label>
                  <input
                    type="number"
                    value={maxLen}
                    min={8}
                    max={ceiling}
                    onChange={(e) => setMaxLen(Number(e.target.value))}
                    className={INPUT}
                  />
                </div>
                <div className="flex items-end">
                  <label className="flex items-center gap-2 pb-2 text-sm">
                    <input
                      type="checkbox"
                      checked={attention}
                      onChange={(e) => setAttention(e.target.checked)}
                    />
                    Capture attention maps
                  </label>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Attention capture grows with the square of sequence length — at 2048 tokens a
                3B model needs over 12&nbsp;GB for a single run. {modeInfo?.limits.note}
              </p>

              {modeInfo?.analyses && (
                <AdvancedOptions
                  analyses={modeInfo.analyses.filter(
                    (a) => a.name !== 'enable_attention_capture',
                  )}
                  mode={mode}
                  values={advanced}
                  onChange={(name, value) =>
                    setAdvanced((prev) => ({ ...prev, [name]: value }))
                  }
                />
              )}

              <button
                type="button"
                onClick={launch}
                disabled={createRun.isPending || !modelA.trim()}
                className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                {createRun.isPending ? 'Starting…' : 'Run analysis'}
              </button>
            </div>
          )}
        </div>

        <div className="rounded-lg border bg-card shadow-sm">
          <h2 className="border-b p-4 text-lg font-semibold">Runs</h2>
          {loadingRuns ? (
            <div className="p-6">
              <LoadingSpinner />
            </div>
          ) : runs.length === 0 ? (
            <p className="p-6 text-sm text-muted-foreground">No analyses yet.</p>
          ) : (
            <table className="w-full text-sm">
              <thead className="border-b text-left text-xs text-muted-foreground">
                <tr>
                  <th className="p-3">#</th>
                  <th className="p-3">Mode</th>
                  <th className="p-3">Model</th>
                  <th className="p-3">Status</th>
                  <th className="p-3">Created</th>
                  <th className="p-3"></th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr
                    key={r.id}
                    onClick={() => router.push(`/interp/${r.id}`)}
                    className="cursor-pointer border-b hover:bg-muted/50"
                  >
                    <td className="p-3">{r.id}</td>
                    <td className="p-3">{r.mode}</td>
                    <td className="p-3 font-mono text-xs">
                      {r.model_a}
                      {r.model_b ? ` vs ${r.model_b}` : ''}
                    </td>
                    <td className="p-3">
                      <StatusBadge status={r.status} />
                    </td>
                    <td className="p-3 text-muted-foreground">
                      {r.created_at ? formatDateTime(r.created_at) : '—'}
                    </td>
                    <td className="p-3">
                      <button
                        type="button"
                        onClick={async (e) => {
                          e.stopPropagation()
                          if (!confirm(`Delete interp run #${r.id} and its artifacts?`)) return
                          try {
                            await deleteRun.mutateAsync(r.id)
                            toast.success(`Deleted run #${r.id}`)
                          } catch (err: any) {
                            toast.error(err?.message || 'Delete failed')
                          }
                        }}
                        className="text-muted-foreground hover:text-destructive"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </Layout>
  )
}
