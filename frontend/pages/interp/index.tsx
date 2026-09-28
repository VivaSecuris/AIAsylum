import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { Brain, Play, Trash2 } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { AdvancedOptions } from '@/components/interp/AdvancedOptions'
import { PatchingOptions, parsePatchOptions, type PatchOptions } from '@/components/interp/PatchingOptions'
import { ModelFitPanel } from '@/components/interp/ModelFitPanel'
import { ModelPicker } from '@/components/models/ModelPicker'
import { useModelCatalog } from '@/lib/model-catalog'
import { parseInterpPrefill } from '@/lib/interp-prefill'
import { getSettings } from '@/lib/settings'
import {
  useInterpModes,
  useInterpRuns,
  useCreateInterpRun,
  useDeleteInterpRun,
} from '@/lib/hooks'
import { apiClient, type InterpMode, type InterpPreflight, type InterpRunRequest } from '@/lib/api'
import { toast } from '@/lib/toast'
import { formatApiError, formatDateTime } from '@/lib/utils'

const INPUT =
  'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

export default function InterpIndexPage() {
  const router = useRouter()
  const { data: modeInfo, isLoading: loadingModes, error: modesError, refetch: reloadModes } = useInterpModes()
  const { data: runs = [], isLoading: loadingRuns, error: runsError } = useInterpRuns({ limit: 50 })
  const createRun = useCreateInterpRun()
  const deleteRun = useDeleteInterpRun()
  const { data: catalog } = useModelCatalog()

  const [mode, setMode] = useState<InterpMode>('single')
  const [modelA, setModelA] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [modelB, setModelB] = useState('')
  const [promptA, setPromptA] = useState('')
  const [promptB, setPromptB] = useState('')
  const [promptList, setPromptList] = useState('')
  const [restoredPrompts, setRestoredPrompts] = useState<string[] | null>(null)
  const [maxLen, setMaxLen] = useState(512)
  const [window, setWindow] = useState(128)
  const [device, setDevice] = useState('auto')
  const [dtype, setDtype] = useState('bfloat16')
  const [topk, setTopk] = useState(10)
  const [dimReduction, setDimReduction] = useState('pca')
  const [prefillWarnings, setPrefillWarnings] = useState<string[]>([])
  const [advanced, setAdvanced] = useState<Record<string, boolean>>({})
  const [patch, setPatch] = useState<PatchOptions>({ component: 'layer', layers: '', positions: '', units: '' })
  const [checking, setChecking] = useState(false)
  const [fit, setFit] = useState<{ key: string; result: InterpPreflight } | null>(null)
  const [fitError, setFitError] = useState<{ key: string; message: string } | null>(null)
  const [launchError, setLaunchError] = useState<string | null>(null)

  const prefillKey = JSON.stringify(['mode', 'model_a', 'model_b', 'prompt', 'prompt_a', 'prompt_b', 'prompts', 'options', 'lineage_parent'].map((key) => router.query[key] ?? null))
  const appliedPrefill = useRef<string | null>(null)
  useEffect(() => {
    if (!router.isReady || appliedPrefill.current === prefillKey) return
    appliedPrefill.current = prefillKey
    const defaults = getSettings()
    const defaultModel = ['transformers', 'local'].includes(defaults.defaultPatientProvider) && defaults.defaultPatientModel
      ? defaults.defaultPatientModel : undefined
    const next = parseInterpPrefill(router.query, defaultModel)
    setMode(next.mode)
    setModelA(next.modelA)
    setModelB(next.modelB)
    setPromptA(next.promptA)
    setPromptB(next.promptB)
    setPromptList(next.prompts.join('\n'))
    setRestoredPrompts(next.prompts.length ? next.prompts : null)
    setMaxLen(next.maxLen)
    setWindow(next.window)
    setTopk(next.topk)
    setDimReduction(next.dimReduction)
    setDevice(next.device)
    setDtype(next.dtype)
    setAdvanced(next.advanced)
    setPatch(next.patch)
    setPrefillWarnings(next.warnings)
    setFit(null)
    setFitError(null)
    setLaunchError(null)
  }, [router.isReady, prefillKey])

  const spec = modeInfo?.modes.find((m) => m.name === mode)
  const needs = (field: string) => spec?.needs.includes(field) ?? false
  const ceiling = modeInfo?.limits.max_len_ceiling ?? 1024
  const prompts = restoredPrompts ?? promptList.split('\n').map((p) => p.trim()).filter(Boolean)
  const allowed = (modeInfo?.analyses ?? []).filter(
    (a) => a.available !== false && (!a.modes || a.modes.includes(mode)),
  )
  const flags = Object.fromEntries(allowed.map((a) => [a.name, !!advanced[a.name]]))
  const skippedAnalyses = modeInfo ? Object.entries(advanced).filter(([name, enabled]) => enabled && !allowed.some((analysis) => analysis.name === name)).map(([name]) => modeInfo.analyses?.find((analysis) => analysis.name === name)?.label ?? name) : []
  const parsedPatch = parsePatchOptions(patch)
  const request: InterpRunRequest = {
    mode,
    model_a: modelA.trim(),
    model_b: needs('model_b') ? modelB.trim() : undefined,
    prompt_a: needs('prompt_a') ? promptA : undefined,
    prompt_b: needs('prompt_b') ? promptB : undefined,
    prompts: needs('prompts') ? prompts : undefined,
    max_len: maxLen,
    window,
    device,
    dtype,
    topk,
    dim_reduction: dimReduction,
    lineage_parent: typeof router.query.lineage_parent === 'string' ? router.query.lineage_parent : undefined,
    ...flags,
    ...(flags.enable_patching ? parsedPatch.request : {}),
  }
  // A result belongs to the exact inputs checked, including captures and prompts.
  // An in-flight response cannot approve inputs edited while it was running.
  const requestKey = JSON.stringify(request)
  const currentFit = fit?.key === requestKey ? fit.result : null
  const currentFitError = fitError?.key === requestKey ? fitError.message : null
  const unavailableModels = [modelA, ...(needs('model_b') ? [modelB] : [])]
    .map((ref) => catalog?.models.find((model) => model.model_ref === ref))
    .filter((model) => model && ['missing', 'incomplete'].includes(model.availability))
  const issues = [
    !spec && 'Analysis capabilities are unavailable.',
    !modelA.trim() && 'Enter a model ID or a path on the analysis server.',
    needs('model_b') && !modelB.trim() && 'Enter the second model.',
    needs('prompt_a') && !promptA.trim() && 'Enter a prompt.',
    needs('prompt_b') && !promptB.trim() && 'Enter the comparison prompt.',
    needs('prompts') && prompts.length < 2 && 'Enter at least two prompts, one per line.',
    needs('prompts') && prompts.some((prompt) => !prompt.trim()) && 'Every prompt must contain text.',
    ...(flags.enable_patching ? parsedPatch.errors : []),
    flags.enable_patching && parsedPatch.request.patch_positions?.some((position) => position >= Math.min(window, maxLen)) && 'Patch positions must fit the analysis window.',
    ...unavailableModels.map((model) => `${model!.name}: ${model!.reason || 'Checkpoint is unavailable on this server.'}`),
    (!Number.isInteger(maxLen) || maxLen < 8 || maxLen > ceiling) && `Sequence length must be an integer from 8 to ${ceiling}.`,
    (!Number.isInteger(window) || window < 1 || window > maxLen) && 'Analysis window must be between 1 and the sequence length.',
    (!Number.isInteger(topk) || topk < 1 || topk > 100) && 'Prediction count must be between 1 and 100.',
  ].filter(Boolean) as string[]

  function selectMode(next: InterpMode) {
    setMode(next)
    const supported = new Set(
      (modeInfo?.analyses ?? [])
        .filter((a) => a.available !== false && (!a.modes || a.modes.includes(next)))
        .map((a) => a.name),
    )
    setAdvanced((previous) => Object.fromEntries(Object.entries(previous).filter(([key]) => supported.has(key))))
    setLaunchError(null)
  }

  async function checkFit() {
    if (issues.length || checking) return
    setChecking(true)
    setFitError(null)
    try {
      setFit({ key: requestKey, result: await apiClient.interpPreflight(request) })
    } catch (error) {
      setFit(null)
      setFitError({ key: requestKey, message: formatApiError(error, 'Could not check model fit') })
    } finally {
      setChecking(false)
    }
  }

  async function launch() {
    if (issues.length || !currentFit?.ready || checking) return
    setLaunchError(null)
    try {
      const run = await createRun.mutateAsync(request)
      toast.success(`Started ${spec?.label.toLowerCase() ?? mode} analysis`)
      router.push(`/interp/${run.id}`)
    } catch (error) {
      const message = formatApiError(error, 'Could not start the analysis')
      setLaunchError(message)
      toast.error(message)
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
              Follow a model’s representations through its layers, compare prompts, and test which activations change a prediction.
            </p>
          </div>
        </div>

        <Link href="/compare?view=models" className="inline-flex text-sm text-primary hover:underline">← Model library</Link>
        {typeof router.query.lineage_parent === 'string' && <p className="rounded-lg border bg-primary/5 p-3 text-sm">This analysis starts a new branch in the history graph. The previous run and its artifacts are preserved.</p>}
        {prefillWarnings.length > 0 && <div role="alert" className="rounded-lg border border-amber-400 bg-amber-50/40 p-3 text-sm dark:bg-amber-950/20">{prefillWarnings.map((warning) => <p key={warning}>{warning}</p>)}</div>}

        <div className="rounded-lg border bg-card p-4 text-sm">
          <p className="font-medium">Runs execute on the connected analysis server</p>
          <p className="mt-1 text-muted-foreground">
            {modeInfo?.hardware?.devices?.length
              ? modeInfo.hardware.devices.map((d) => `${d.name} (${d.device})${d.free_gb != null ? ` · ${d.free_gb.toFixed(1)} GiB free` : ''}`).join(' / ')
              : 'Check model fit to inspect the server and estimate memory before loading weights.'}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Model paths and downloaded weights belong to that server. Selecting CUDA uses a GPU already attached to it.
          </p>
        </div>

        {(modeInfo?.warnings?.length ?? 0) > 0 && (
          <div className="rounded-lg border border-yellow-500/50 bg-yellow-50 p-4 dark:bg-yellow-950">
            <p className="mb-2 text-sm font-medium">Server notes</p>
            <ul className="list-inside list-disc space-y-1 text-sm text-muted-foreground">
              {modeInfo!.warnings.map((warning, i) => <li key={i}>{warning}</li>)}
            </ul>
          </div>
        )}

        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h2 className="mb-2 text-lg font-semibold">{mode === 'model_diff' ? 'Compare a custom model with its original' : 'New analysis'}</h2>
          <p className="mb-4 text-sm text-muted-foreground">Choose models, add a prompt, then check fit before starting the analysis.</p>
          {loadingModes ? <LoadingSpinner /> : modesError ? (
            <div role="alert" className="space-y-2 text-sm">
              <p>{formatApiError(modesError, 'Could not load analysis capabilities')}</p>
              <button onClick={() => reloadModes()} className="rounded border px-3 py-2">Retry connection</button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label htmlFor="interp-mode" className="text-xs text-muted-foreground">Mode</label>
                <select id="interp-mode" value={mode} onChange={(e) => selectMode(e.target.value as InterpMode)} className={INPUT}>
                  {modeInfo?.modes.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}
                </select>
                <p className="mt-1 text-xs text-muted-foreground">{spec?.description}</p>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <ModelPicker id="interp-model-a" label={needs('model_b') ? 'Original model (baseline)' : 'Model'} value={modelA} onChange={setModelA} />
                {needs('model_b') && (
                  <ModelPicker id="interp-model-b" label="Custom model" value={modelB} onChange={(ref) => {
                    setModelB(ref)
                    const original = catalog?.models.find((model) => model.model_ref === ref)?.source_model
                    if (original) setModelA(original)
                  }} />
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Hugging Face weights download on first run and remain cached on the server. Use a small model to validate an analysis before scaling up.
                {mode === 'model_diff' && ' Direct model comparison requires the same tokenizer, layer count and hidden size; analyze different model sizes in separate runs.'}
              </p>

              {needs('prompt_a') && (
                <div>
                  <label htmlFor="interp-prompt-a" className="text-xs text-muted-foreground">{needs('prompt_b') ? 'Prompt A (reference)' : 'Prompt'}</label>
                  <textarea id="interp-prompt-a" value={promptA} onChange={(e) => setPromptA(e.target.value)} rows={3} className={INPUT} placeholder="The capital of France is" />
                </div>
              )}
              {needs('prompt_b') && (
                <div>
                  <label htmlFor="interp-prompt-b" className="text-xs text-muted-foreground">Prompt B (comparison)</label>
                  <textarea id="interp-prompt-b" value={promptB} onChange={(e) => setPromptB(e.target.value)} rows={3} className={INPUT} placeholder="The capital of Germany is" />
                  <p className="mt-1 text-xs text-muted-foreground">Change one feature of the prompt to make differences easier to interpret.</p>
                </div>
              )}
              {needs('prompts') && (
                <div>
                  {restoredPrompts ? <div className="space-y-3">
                    <p className="text-xs text-muted-foreground">Saved prompts in order (zero-shot first). Line breaks inside each prompt are preserved.</p>
                    {restoredPrompts.map((prompt, index) => <div key={index}><div className="flex items-center justify-between"><label htmlFor={`interp-restored-prompt-${index}`} className="text-xs text-muted-foreground">Prompt {index + 1}</label>{restoredPrompts.length > 2 && <button type="button" onClick={() => setRestoredPrompts((previous) => previous!.filter((_, item) => item !== index))} className="text-xs text-muted-foreground hover:text-foreground">Remove prompt {index + 1}</button>}</div><textarea id={`interp-restored-prompt-${index}`} value={prompt} onChange={(event) => setRestoredPrompts((previous) => previous!.map((value, item) => item === index ? event.target.value : value))} rows={3} className={INPUT} /></div>)}
                    <button type="button" onClick={() => setRestoredPrompts((previous) => [...previous!, ''])} className="rounded-md border px-3 py-2 text-xs hover:bg-muted">Add a prompt</button>
                  </div> : <><label htmlFor="interp-prompts" className="text-xs text-muted-foreground">Prompts, one per line (zero-shot first)</label><textarea id="interp-prompts" value={promptList} onChange={(e) => setPromptList(e.target.value)} rows={4} className={INPUT} /></>}
                  <p className="mt-1 text-xs text-muted-foreground">{prompts.length} prompts; at least two required.</p>
                </div>
              )}

              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <label htmlFor="interp-device" className="text-xs text-muted-foreground">Compute device</label>
                  <select id="interp-device" value={device} onChange={(e) => setDevice(e.target.value)} className={INPUT}>
                    <option value="auto">Automatic ({modeInfo?.hardware?.default_device ?? 'server default'})</option>
                    <option value="cuda">CUDA GPU</option>
                    {(modeInfo?.hardware?.devices ?? []).filter((d) => d.device.startsWith('cuda:')).map((d) => <option key={d.device} value={d.device}>{d.device} · {d.name}</option>)}
                    {device.startsWith('cuda:') && !modeInfo?.hardware?.devices.some((item) => item.device === device) && <option value={device}>{device} · saved device (not currently detected)</option>}
                    <option value="mps">Apple GPU (MPS)</option>
                    <option value="cpu">CPU</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="interp-dtype" className="text-xs text-muted-foreground">Weight precision</label>
                  <select id="interp-dtype" value={dtype} onChange={(e) => setDtype(e.target.value)} className={INPUT}>
                    <option value="bfloat16">BF16 · 2 bytes / parameter</option>
                    <option value="float16">FP16 · 2 bytes / parameter</option>
                    <option value="float32">FP32 · 4 bytes / parameter</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="interp-max-len" className="text-xs text-muted-foreground">Sequence limit (max {ceiling})</label>
                  <input id="interp-max-len" type="number" value={maxLen} min={8} max={ceiling} onChange={(e) => setMaxLen(Number(e.target.value))} className={INPUT} />
                </div>
                <div>
                  <label htmlFor="interp-window" className="text-xs text-muted-foreground">Analysis window (tokens)</label>
                  <input id="interp-window" type="number" value={window} min={1} max={maxLen} onChange={(e) => setWindow(Number(e.target.value))} className={INPUT} />
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                The sequence limit truncates model input. The window controls the tokens shown in analysis; reducing it does not reduce the full forward-pass cost. Attention capture grows quadratically with sequence length.
              </p>
              <details className="rounded-md border p-3"><summary className="cursor-pointer text-sm">Prediction and projection settings</summary><div className="mt-3 grid gap-4 sm:grid-cols-2"><div><label htmlFor="interp-topk" className="text-xs text-muted-foreground">Predictions to show per layer</label><input id="interp-topk" type="number" min={1} max={100} value={topk} onChange={(event) => setTopk(Number(event.target.value))} className={INPUT} /></div><div><label htmlFor="interp-projection" className="text-xs text-muted-foreground">Representation projection</label><select id="interp-projection" value={dimReduction} onChange={(event) => setDimReduction(event.target.value)} className={INPUT}><option value="pca">PCA</option><option value="tsne">t-SNE</option><option value="umap">UMAP</option></select></div></div></details>

              {modeInfo?.analyses && <AdvancedOptions analyses={modeInfo.analyses} mode={mode} values={advanced} onChange={(name, value) => setAdvanced((previous) => ({ ...previous, [name]: value }))} />}
              {flags.enable_patching && <PatchingOptions value={patch} onChange={setPatch} />}
              {skippedAnalyses.length > 0 && <p className="text-xs text-amber-700 dark:text-amber-300">Saved analyses unavailable in this mode will not run: {skippedAnalyses.join(', ')}.</p>}

              {!!issues.length && <p className="text-xs text-muted-foreground">{issues.join(' ')}</p>}
              {currentFit && <ModelFitPanel result={currentFit} />}
              {!currentFit && fit && <p className="text-sm text-amber-700 dark:text-amber-300">Inputs changed. Check model fit again before running.</p>}
              {(currentFitError || launchError) && <p role="alert" className="rounded border border-destructive/50 p-3 text-sm text-destructive">{currentFitError || launchError}</p>}

              <div className="flex flex-wrap gap-3">
                <button type="button" onClick={checkFit} disabled={checking || createRun.isPending || !!issues.length} className="rounded-md border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50">
                  {checking ? 'Checking server and model configuration…' : 'Check model fit'}
                </button>
                <button type="button" onClick={launch} disabled={checking || createRun.isPending || !!issues.length || !currentFit?.ready} className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50">
                  <Play className="h-4 w-4" />{createRun.isPending ? 'Starting…' : 'Run analysis'}
                </button>
              </div>
              <p className="text-xs text-muted-foreground">Fit checks read model configuration without downloading weights. Memory estimates are approximate; the server checks again at launch.</p>
            </div>
          )}
        </div>

        <div className="rounded-lg border bg-card shadow-sm">
          <h2 className="border-b p-4 text-lg font-semibold">Runs</h2>
          {runsError ? (
            <p role="alert" className="p-6 text-sm text-destructive">{formatApiError(runsError, 'Could not load analyses')}</p>
          ) : loadingRuns ? (
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
                            toast.error(formatApiError(err, 'Delete failed'))
                          }
                        }}
                        aria-label={`Delete analysis ${r.id}`}
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
