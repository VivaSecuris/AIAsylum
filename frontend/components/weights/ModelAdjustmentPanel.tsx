import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { apiClient, type PreflightResponse, type ProviderInfo } from '@/lib/api'
import { getSettings } from '@/lib/settings'
import { chatErrorMessage, chatSettingsFromDefaults, supportsWeightActions, type ChatExchange, type ChatResponse, type ChatSettings } from '@/lib/model-chat'
import { adjustmentCanApply, adjustmentCanUse, adjustmentPreflightParams, buildAdjustmentRequest, type AdjustmentCoach, type AdjustmentMode, type ModelAdjustment } from '@/lib/model-adjustments'
import { ModelSelector } from '@/components/forms/ModelSelector'
import { PreflightPanel } from '@/components/weights/PreflightPanel'
import { TurnSystemPrompts } from '@/components/conversation/ConversationViewer'

const INPUT = 'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm disabled:opacity-50'
const BUTTON = 'rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50'

/** Both sides retain their actual system messages above requested reasoning and the answer. */
export function AdjustmentResponse({ title, response }: { title: string; response: ChatResponse | string }) {
  if (typeof response === 'string') return <article className="space-y-2 rounded border p-3"><h4 className="text-sm font-semibold">{title}</h4><p className="whitespace-pre-wrap break-words text-sm">{response}</p><p className="text-xs text-muted-foreground">Saved answer text; generation evidence was not recorded with this baseline.</p></article>
  const reasoning = response.reasoning || response.metadata?.reasoning
  const native = response.metadata?.native_reasoning
  return <article className="space-y-2 rounded border p-3">
    <h4 className="text-sm font-semibold">{title}</h4><p className="break-words text-xs text-muted-foreground">{response.provider} / {response.model}</p>
    <TurnSystemPrompts prompts={response.metadata?.request_system_prompts} />
    {typeof reasoning === 'string' && reasoning && <details className="text-xs"><summary className="cursor-pointer">{(response.reasoning_source || response.metadata?.reasoning_source) === 'react' ? 'Requested reasoning (ReACT)' : 'Recorded reasoning'}</summary><p className="mt-2 whitespace-pre-wrap break-words">{reasoning}</p></details>}
    {typeof native === 'string' && native && <details className="text-xs"><summary className="cursor-pointer">Model-provided reasoning</summary><p className="mt-2 whitespace-pre-wrap break-words">{native}</p></details>}
    <p className="whitespace-pre-wrap break-words text-sm">{response.content || '(The model returned no answer text.)'}</p>
    {response.finish_reason === 'length' && <p className="text-xs text-amber-700 dark:text-amber-300">Output reached its token limit.</p>}
    <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">Generation evidence &amp; token usage</summary><pre className="mt-2 overflow-auto whitespace-pre-wrap break-words">{JSON.stringify({ generation: response.generation, finish_reason: response.finish_reason, usage: response.usage }, null, 2)}</pre></details>
  </article>
}

export function ModelAdjustmentPanel({ provider, model, settings, exchanges, parentId, disabled, onUseVersion, onBusyChange }: {
  provider: string
  model: string
  settings: ChatSettings
  exchanges: ChatExchange[]
  parentId?: string
  disabled?: boolean
  onUseVersion: (adjustment: ModelAdjustment) => void
  onBusyChange?: (busy: boolean) => void
}) {
  const [instruction, setInstruction] = useState('')
  const [mode, setMode] = useState<AdjustmentMode>('profile')
  const [coach, setCoach] = useState<AdjustmentCoach>(() => {
    const saved = getSettings()
    const generation = chatSettingsFromDefaults(saved, 'doctor')
    return { provider: saved.defaultDoctorProvider, model: saved.defaultDoctorModel, temperature: generation.temperature, top_p: generation.top_p, max_tokens: generation.max_tokens, enable_cot: generation.enable_cot, seed: settings.seed }
  })
  const [revisions, setRevisions] = useState<ModelAdjustment[]>([])
  const [revisionIssue, setRevisionIssue] = useState('')
  const [providers, setProviders] = useState<ProviderInfo[] | null>(null)
  const [providerIssue, setProviderIssue] = useState('')
  const [adjustment, setAdjustment] = useState<ModelAdjustment | null>(null)
  const [preflight, setPreflight] = useState<PreflightResponse | null>(null)
  const [preflightError, setPreflightError] = useState('')
  const [preflightRefresh, setPreflightRefresh] = useState(0)
  const [acknowledged, setAcknowledged] = useState<string[]>([])
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const requestVersion = useRef(0)
  const inFlight = useRef(false)
  const localWeights = supportsWeightActions(provider)
  const hasReply = !!exchanges.at(-1)?.response.content.trim()
  const providerInfo = (name: string) => providers?.find((item) => item.name === name || item.aliases.includes(name))
  const targetInfo = providerInfo(provider)
  const coachInfo = providerInfo(coach.provider)
  const targetIssue = providerIssue || (!providers ? 'Checking adjustment support…' : !targetInfo ? 'This provider is not recognized by the connected server.' : targetInfo.kind === 'service' ? 'This provider does not accept the full conversation and system instructions required by adjustments. Its ordinary chat remains available.' : '')
  const coachIssue = providers && coach.provider ? !coachInfo ? 'Choose a supported coach provider.' : coachInfo.kind === 'service' ? 'This coach provider does not accept the system instructions and conversation required to propose adjustments.' : '' : ''

  useEffect(() => {
    onBusyChange?.(!!busy)
    return () => { onBusyChange?.(false) }
  }, [busy, onBusyChange])

  useEffect(() => {
    let cancelled = false
    apiClient.listProviders().then((items) => { if (!cancelled) { setProviders(items); setProviderIssue('') } })
      .catch(() => { if (!cancelled) setProviderIssue('Provider support could not be checked. Reload the page before requesting an adjustment.') })
    apiClient.listModelAdjustments({ provider, model }).then((items) => {
      if (!cancelled) { setRevisions(items); setRevisionIssue('') }
    }).catch((cause) => { if (!cancelled) setRevisionIssue(chatErrorMessage(cause)) })
    return () => { cancelled = true; requestVersion.current += 1 }
  }, [provider, model])

  useEffect(() => {
    setPreflight(null); setPreflightError(''); setAcknowledged([])
    if (adjustment?.mode !== 'weights' || adjustment.status !== 'proposed' || !adjustment.weight_request) return
    let cancelled = false
    apiClient.weightPreflight(adjustmentPreflightParams(adjustment.weight_request)).then((result) => {
      if (!cancelled) setPreflight(result)
    }).catch((cause) => { if (!cancelled) setPreflightError(chatErrorMessage(cause)) })
    return () => { cancelled = true }
  }, [adjustment?.id, adjustment?.status, preflightRefresh])

  useEffect(() => {
    if (adjustment?.status !== 'training' && adjustment?.status !== 'applying') return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const refresh = async () => {
      try {
        const current = await apiClient.getModelAdjustment(adjustment.id)
        if (cancelled) return
        setAdjustment(current)
        if (current.status === 'training' || current.status === 'applying') timer = setTimeout(refresh, 3000)
      } catch (cause) {
        if (!cancelled) { setError(chatErrorMessage(cause)); timer = setTimeout(refresh, 10000) }
      }
    }
    timer = setTimeout(refresh, 3000)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [adjustment?.id, adjustment?.status])

  async function operate(label: string, action: (version: number) => Promise<void>) {
    if (inFlight.current || disabled) return
    inFlight.current = true
    const version = ++requestVersion.current
    setBusy(label); setError('')
    try { await action(version) }
    catch (cause) { if (version === requestVersion.current) setError(chatErrorMessage(cause)) }
    finally {
      if (version === requestVersion.current) { setBusy(''); inFlight.current = false }
    }
  }

  function remember(value: ModelAdjustment) {
    setAdjustment(value)
    setRevisions((previous) => [value, ...previous.filter((item) => item.id !== value.id)])
  }

  async function propose() {
    if (targetIssue || coachIssue) return
    await operate('Asking the coach for a proposal…', async (version) => {
      const request = buildAdjustmentRequest({ provider, model, instruction, mode, settings, exchanges, coach, ...(parentId ? { parent_id: parentId } : {}) })
      const result = await apiClient.proposeModelAdjustment(request)
      if (version === requestVersion.current) remember(result)
    })
  }

  async function retest(value: ModelAdjustment, version: number) {
    setBusy('Re-testing the saved conversation…')
    const response = await apiClient.testModelAdjustment(value.id)
    if (version !== requestVersion.current) return
    // Show the completed answer immediately; the endpoint also persists it.
    remember({ ...value, tests: [...value.tests, { at: new Date().toISOString(), response, ...(value.active_target ? { target: value.active_target } : {}) }] })
  }

  async function apply() {
    if (targetIssue || !adjustment || !adjustmentCanApply(adjustment, preflight, acknowledged)) return
    await operate(adjustment.mode === 'profile' ? 'Saving the prompt profile…' : 'Starting checkpoint training…', async (version) => {
      const result = await apiClient.applyModelAdjustment(adjustment.id, { acknowledge: acknowledged })
      if (version !== requestVersion.current) return
      remember(result)
      if (result.mode === 'profile' && adjustmentCanUse(result)) await retest(result, version)
    })
  }

  const baseline = adjustment?.baseline.evidence || adjustment?.baseline.response || adjustment?.baseline.messages.at(-1)?.content
  const canApply = !!adjustment && adjustmentCanApply(adjustment, preflight, acknowledged)
  const canUse = !!adjustment && adjustmentCanUse(adjustment)
  const blocked = !!busy || !!disabled

  return <details className="rounded-lg border bg-muted/10 p-4" aria-labelledby="model-adjustment-title">
    <summary id="model-adjustment-title" className="cursor-pointer font-semibold">Adjust this model’s behavior · saved versions</summary>
    <div className="mt-4 space-y-4">
    <p className="text-sm text-muted-foreground">Describe a change to the latest answer. A coach proposes a version for you to review, apply and re-test against the saved conversation.</p>
    {targetIssue && <p role="status" className="text-sm text-muted-foreground">{targetIssue}</p>}
    <fieldset disabled={blocked} className="space-y-3">
      <label className="block text-sm">Describe the behavior change<textarea rows={3} maxLength={8000} value={instruction} onChange={(event) => { setInstruction(event.target.value); setAdjustment(null) }} className={INPUT} placeholder="For example: state uncertainty, check the arithmetic, and avoid inventing credentials." /></label>
      <label className="block text-sm">Adjustment type<select value={mode} onChange={(event) => { setMode(event.target.value as AdjustmentMode); setAdjustment(null) }} className={INPUT}><option value="profile">Saved prompt profile</option><option value="weights" disabled={!localWeights}>Train a new checkpoint (local Transformers)</option></select></label>
      <p className="text-xs text-muted-foreground">{mode === 'profile' ? 'A prompt profile changes the system instructions used with this model. It does not change its weights.' : 'Training creates a separate checkpoint from reviewed examples. The original weights remain available. A few examples can overfit; the re-test alone does not establish a general improvement.'}</p>
      {!localWeights && <p className="text-xs text-muted-foreground">This provider supports prompt profiles. Training needs a local Transformers checkpoint; <Link href="/compare?view=find" className="text-primary underline">find an open-weight version</Link> if one is available.</p>}
      <details className="rounded border p-3"><summary className="cursor-pointer text-sm">Coach model · {coach.provider} / {coach.model || 'choose a model'}</summary><div className="mt-3 space-y-3">
        <p className="text-xs text-muted-foreground">Starts from saved doctor model and generation defaults. The coach receives this conversation, its system prompt and your adjustment instruction to draft the proposal.</p>
        <ModelSelector label="Adjustment coach" provider={coach.provider} model={coach.model} onProviderChange={(provider) => { setCoach((old) => ({ ...old, provider, model: '' })); setAdjustment(null) }} onModelChange={(model) => { setCoach((old) => ({ ...old, model })); setAdjustment(null) }} />
        {coachIssue && <p role="alert" className="text-sm text-destructive">{coachIssue}</p>}
        <label className="flex items-start gap-2 text-sm"><input type="checkbox" checked={!!coach.enable_cot} onChange={(event) => { setCoach((old) => ({ ...old, enable_cot: event.target.checked })); setAdjustment(null) }} /><span>Request ReACT reasoning from the coach<span className="block text-xs text-muted-foreground">Requests extra reasoning text and can change the proposal; it does not expose internal computation or execute actions.</span></span></label>
        <label className="block text-sm">Coach maximum output tokens<input type="number" min={1} max={32768} value={coach.max_tokens ?? ''} placeholder="Provider default" onChange={(event) => { setCoach((old) => ({ ...old, max_tokens: event.target.value === '' ? null : Number(event.target.value) })); setAdjustment(null) }} className={INPUT} /></label>
      </div></details>
      <button type="button" onClick={propose} disabled={blocked || !!targetIssue || !!coachIssue || !hasReply || !instruction.trim() || !coach.provider || !coach.model} className={BUTTON}>Propose adjustment</button>
      {!hasReply && <p className="text-xs text-muted-foreground">Send a message and receive an answer first, or open a saved revision below.</p>}
    </fieldset>
    <label className="block text-sm">Saved revisions<select value={adjustment?.id || ''} disabled={blocked} onChange={(event) => {
      const id = event.target.value
      if (!id) { setAdjustment(null); return }
      void operate('Loading saved revision…', async (version) => {
        const current = await apiClient.getModelAdjustment(id)
        if (version === requestVersion.current) { remember(current); setInstruction(current.instruction); setMode(current.mode) }
      })
    }} className={INPUT}><option value="">Choose a saved revision</option>{revisions.map((item) => <option key={item.id} value={item.id}>{item.mode === 'profile' ? 'Prompt profile' : 'Checkpoint'} · {item.status} · {item.instruction.slice(0, 80)}</option>)}</select></label>
    {revisionIssue && <p role="status" className="text-xs text-muted-foreground">Saved revisions could not be loaded: {revisionIssue}</p>}
    {busy && <p role="status" className="text-sm">{busy}</p>}
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    {adjustment && <div className="space-y-4 rounded border bg-background p-4">
      <div><h4 className="font-medium">{adjustment.mode === 'profile' ? 'Prompt profile' : 'Checkpoint training'} · {adjustment.status}</h4><p className="mt-1 text-sm">{adjustment.proposal.summary}</p><p className="mt-1 break-words text-xs text-muted-foreground">Source: {adjustment.provider} / {adjustment.model} · Revision {adjustment.id}</p></div>
      <div className="grid gap-3 lg:grid-cols-2"><div><h5 className="text-xs font-semibold">Original system prompt</h5><pre className="mt-1 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded border p-3 text-xs">{adjustment.settings.system_prompt || '(No app system prompt)'}</pre></div><div><h5 className="text-xs font-semibold">{adjustment.mode === 'profile' ? 'Proposed system prompt' : 'Suggested profile prompt (not used by training)'}</h5><pre className="mt-1 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded border p-3 text-xs">{adjustment.proposal.system_prompt || '(No app system prompt)'}</pre></div></div>
      {adjustment.mode === 'weights' && <p className="text-xs text-muted-foreground">Training and re-tests keep the original system prompt above so the comparison measures the checkpoint change.</p>}
      {adjustment.coach_evidence && <details className="rounded border p-3"><summary className="cursor-pointer text-sm">Coach generation evidence</summary><div className="mt-2"><AdjustmentResponse title="Coach proposal" response={adjustment.coach_evidence} /></div></details>}
      {adjustment.mode === 'weights' && <details open className="rounded border p-3"><summary className="cursor-pointer text-sm font-medium">Training examples ({adjustment.proposal.training_examples.length})</summary><div className="mt-2 space-y-3">{adjustment.proposal.training_examples.map((example, index) => <div key={index} className="space-y-1 text-sm"><p className="whitespace-pre-wrap break-words"><strong>Input:</strong> {example.prompt}</p><p className="whitespace-pre-wrap break-words"><strong>Desired answer:</strong> {example.response}</p></div>)}</div></details>}
      {!!adjustment.proposal.test_prompts.length && <details className="text-sm"><summary className="cursor-pointer">Suggested follow-up checks</summary><ul className="mt-2 list-inside list-disc space-y-1">{adjustment.proposal.test_prompts.map((prompt, index) => <li key={index} className="whitespace-pre-wrap break-words">{prompt}</li>)}</ul><p className="mt-2 text-xs text-muted-foreground">Suggestions only; these checks have not been run.</p></details>}
      {adjustment.mode === 'weights' && adjustment.status === 'proposed' && <div className="space-y-2">
        {adjustment.weight_request ? <details className="text-xs"><summary className="cursor-pointer">Checkpoint training configuration</summary><pre className="mt-2 overflow-auto whitespace-pre-wrap break-words">{JSON.stringify(adjustment.weight_request, null, 2)}</pre></details> : <p role="alert" className="text-sm text-destructive">The proposal has no executable training request.</p>}
        {preflightError ? <p role="alert" className="text-sm text-destructive">Preflight failed: {preflightError}</p> : !preflight && adjustment.weight_request ? <p role="status" className="text-sm">Checking resources before training…</p> : null}
        {preflight && <PreflightPanel checks={preflight.checks} acknowledged={acknowledged} onAcknowledge={setAcknowledged} />}
        <button type="button" className={BUTTON} disabled={blocked} onClick={() => setPreflightRefresh((count) => count + 1)}>Refresh resource checks</button>
      </div>}
      {adjustment.status === 'proposed' && <button type="button" className={BUTTON} disabled={blocked || !!targetIssue || !canApply} onClick={apply}>{adjustment.mode === 'profile' ? 'Apply profile and re-test' : 'Train new checkpoint'}</button>}
      {adjustment.weight_run && <p className="text-sm"><Link href={`/weights/${adjustment.weight_run.id}`} className="text-primary underline">Training run #{adjustment.weight_run.id}</Link> · {adjustment.weight_run.status}{adjustment.weight_run.error ? ` · ${adjustment.weight_run.error}` : ''}</p>}
      {adjustment.status === 'training' && <p role="status" className="text-sm">Training is in progress. This panel checks for the completed checkpoint; you can return to this saved revision later.</p>}
      {adjustment.status === 'applying' && <p role="status" className="text-sm">This version is being applied. Its saved status will update here.</p>}
      {adjustment.status === 'failed' && <p role="alert" className="text-sm text-destructive">{adjustment.error || 'This adjustment failed. Review the training run for details.'}</p>}
      {canUse && <div className="flex flex-wrap gap-2"><button type="button" className={BUTTON} disabled={blocked} onClick={() => void operate('Re-testing the saved conversation…', (version) => retest(adjustment, version))}>Re-test saved conversation</button><button type="button" className={BUTTON} disabled={blocked} onClick={() => onUseVersion(adjustment)}>Use version in a new chat</button></div>}
      <p className="text-xs text-muted-foreground">Re-tests reuse the saved conversation up to the last question and replace its final answer. Applying a version keeps this chat’s baseline intact. “Use version” starts fresh context without changing global defaults.</p>
      {baseline && <AdjustmentResponse title="Before · saved answer" response={baseline} />}
      {adjustment.baseline.evidence_source === 'client_reported' && <p className="text-xs text-muted-foreground">The baseline generation evidence was supplied by this chat page when the revision was saved.</p>}
      {adjustment.tests.map((test, index) => <AdjustmentResponse key={index} title={`After · re-test ${index + 1}`} response={test.response} />)}
    </div>}
    </div>
  </details>
}
