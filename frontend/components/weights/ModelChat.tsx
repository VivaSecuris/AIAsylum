import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { apiClient, type Prompt } from '@/lib/api'
import { getSettings, saveSettings, type GenerationRole } from '@/lib/settings'
import { buildChatRequest, chatDefaultPrompt, chatErrorMessage, chatSettingsFromDefaults, chatSettingsForHandoff, modelActionUrls, settingsWithChatDefaults, supportsWeightActions, validateChatSettings, type ChatExchange, type ChatSettings } from '@/lib/model-chat'
import { TurnSystemPrompts } from '@/components/conversation/ConversationViewer'
import { ModelAdjustmentPanel } from '@/components/weights/ModelAdjustmentPanel'
import { adjustmentCanUse, type ModelAdjustment } from '@/lib/model-adjustments'

const INPUT = 'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm disabled:opacity-50'
const BUTTON = 'rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50'
const numberValue = (value: string) => value === '' ? null : Number(value)

export function ModelChat({ provider, model, name = model, sourceModel, disabledReason, initialAdjustmentId }: {
  provider: string; model: string; name?: string; sourceModel?: string | null; disabledReason?: string; initialAdjustmentId?: string
}) {
  const [exchanges, setExchanges] = useState<ChatExchange[]>([])
  const [draft, setDraft] = useState('')
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [settings, setSettings] = useState<ChatSettings | null>(null)
  const [loadingDefaults, setLoadingDefaults] = useState(true)
  const [defaultPromptIssue, setDefaultPromptIssue] = useState<string | null>(null)
  const [savedMessage, setSavedMessage] = useState('')
  const [promptRole, setPromptRole] = useState<GenerationRole>('patient')
  const [presets, setPresets] = useState<Prompt[]>([])
  const [presetId, setPresetId] = useState('')
  const [presetIssue, setPresetIssue] = useState<string | null>(null)
  const [promptSource, setPromptSource] = useState('Saved patient defaults')
  const [activeRevisionId, setActiveRevisionId] = useState<string | undefined>()
  const [restoringRevision, setRestoringRevision] = useState(false)
  const [adjusting, setAdjusting] = useState(false)
  const restoredRevision = useRef('')
  const requestVersion = useRef(0)
  const end = useRef<HTMLDivElement>(null)
  const busy = pendingPrompt !== null
  const localWeights = supportsWeightActions(provider)

  useEffect(() => {
    let cancelled = false
    const defaults = getSettings()
    const initial = chatSettingsFromDefaults(defaults, promptRole)
    const savedPrompt = chatDefaultPrompt(defaults, promptRole)
    requestVersion.current += 1
    setExchanges([]); setPendingPrompt(null); setError(null); setSavedMessage('')
    setSettings(initial); setLoadingDefaults(true); setDefaultPromptIssue(null)
    setPresets([]); setPresetId(''); setPresetIssue(null)
    setPromptSource(`Saved ${promptRole} defaults`)
    setActiveRevisionId(undefined)
    apiClient.listPrompts({ prompt_type: 'system_prompt', target: promptRole, limit: 1000 }).then((items) => {
      if (!cancelled) setPresets(items.filter((prompt) => prompt.prompt_type === 'system_prompt' && prompt.target === promptRole))
    }).catch(() => { if (!cancelled) setPresetIssue('System presets could not be loaded. You can still write custom instructions.') })
    if (savedPrompt.id) {
      apiClient.getPrompt(savedPrompt.id).then((prompt) => {
        if (cancelled) return
        if (prompt.prompt_type !== 'system_prompt' || prompt.target !== promptRole) {
          setDefaultPromptIssue(`The saved ${promptRole} prompt belongs to a different role. Choose system instructions below before sending.`)
        } else {
          setSettings({ ...initial, system_prompt: prompt.prompt_text })
          setPromptSource(`Saved ${promptRole} preset: ${prompt.name}`)
          setPresetId(String(prompt.id))
        }
      }).catch(() => {
        if (!cancelled) setDefaultPromptIssue(`The saved ${promptRole} prompt could not be loaded. Choose system instructions below before sending.`)
      }).finally(() => { if (!cancelled) setLoadingDefaults(false) })
    } else setLoadingDefaults(false)
    return () => { cancelled = true }
  }, [promptRole])

  useEffect(() => {
    setExchanges([]); setDraft(''); setError(null); setPendingPrompt(null)
    setActiveRevisionId(undefined)
    requestVersion.current += 1
    return () => { requestVersion.current += 1 }
  }, [provider, model])
  useEffect(() => {
    if (!initialAdjustmentId || loadingDefaults) return
    const revisionKey = JSON.stringify([provider, model, initialAdjustmentId])
    if (restoredRevision.current === revisionKey) return
    restoredRevision.current = revisionKey
    let cancelled = false
    requestVersion.current += 1
    setPendingPrompt(null)
    setRestoringRevision(true)
    apiClient.getModelAdjustment(initialAdjustmentId).then((revision) => {
      if (cancelled) return
      const target = revision.active_target
      if (!adjustmentCanUse(revision) || !target || target.provider !== provider || target.model !== model) throw new Error('This saved version is not ready for the selected model.')
      validateChatSettings(target.settings)
      setSettings(target.settings); setExchanges([]); setDraft(''); setPendingPrompt(null)
      setActiveRevisionId(revision.id); setDefaultPromptIssue(null); setPresetId('')
      setPromptSource(`Saved ${revision.mode === 'profile' ? 'prompt profile' : 'checkpoint revision'} ${revision.id}`)
    }).catch((cause) => { if (!cancelled) setDefaultPromptIssue(chatErrorMessage(cause)) })
      .finally(() => { if (!cancelled) setRestoringRevision(false) })
    return () => { cancelled = true }
  }, [initialAdjustmentId, provider, model, loadingDefaults])
  useEffect(() => { end.current?.scrollIntoView({ block: 'nearest' }) }, [exchanges.length, busy])

  function updateSettings(patch: Partial<ChatSettings>) {
    requestVersion.current += 1
    setSettings((previous) => previous && ({ ...previous, ...patch }))
    setExchanges([]); setError(null); setSavedMessage('')
    setActiveRevisionId(undefined)
    if ('system_prompt' in patch) {
      setDefaultPromptIssue(null); setPresetId('')
      setPromptSource(patch.system_prompt ? 'Custom instructions' : 'No app system prompt')
    }
  }

  function useAdjustment(revision: ModelAdjustment) {
    if (!adjustmentCanUse(revision) || !revision.active_target) return
    const target = revision.active_target
    if (target.provider !== provider || target.model !== model) {
      window.location.assign('/models/chat?' + new URLSearchParams({ provider: target.provider, model: target.model, adjustment_id: revision.id }).toString())
      return
    }
    updateSettings(target.settings)
    setDraft(''); setPendingPrompt(null); setActiveRevisionId(revision.id)
    setPromptSource(`Saved prompt profile ${revision.id}`)
  }

  function reset() {
    requestVersion.current += 1
    setExchanges([]); setDraft(''); setError(null)
  }

  function saveDefaults() {
    if (!settings) return
    try {
      saveSettings(settingsWithChatDefaults(getSettings(), settings, promptRole))
      setSavedMessage(`Saved as ${promptRole} defaults for new chats and tests.`)
      setError(null)
    } catch (cause) { setError(chatErrorMessage(cause)) }
  }

  function exportTranscript() {
    const payload = { model, provider, exported_at: new Date().toISOString(), settings, exchanges }
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${name.split('/').pop() || 'model'}-chat-${new Date().toISOString().slice(0, 10)}.json`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  async function send(event: React.FormEvent) {
    event.preventDefault()
    if (busy || adjusting || !settings || loadingDefaults || restoringRevision || defaultPromptIssue || disabledReason) return
    setError(null)
    const prompt = draft.trim()
    let request
    try { request = buildChatRequest(exchanges, prompt, settings) } catch (cause) { setError(chatErrorMessage(cause)); return }
    const version = ++requestVersion.current
    setPendingPrompt(prompt)
    try {
      const response = await apiClient.chatWithAnyModel({ provider, model, ...request })
      if (requestVersion.current !== version) return
      setExchanges((previous) => [...previous, { prompt, response, at: new Date().toISOString() }])
      setDraft('')
    } catch (cause) {
      if (requestVersion.current === version) setError(chatErrorMessage(cause))
    } finally {
      if (requestVersion.current === version) setPendingPrompt(null)
    }
  }

  let actions: ReturnType<typeof modelActionUrls> | undefined
  let actionError = ''
  if (settings && !loadingDefaults && !defaultPromptIssue) {
    try { actions = modelActionUrls(provider, model, chatSettingsForHandoff(settings, exchanges.at(-1)?.response), draft || exchanges.at(-1)?.prompt || '', getSettings(), sourceModel) }
    catch (cause) { actionError = chatErrorMessage(cause) }
  }

  return <section className="space-y-4 rounded-lg border bg-card p-4 sm:p-6" aria-labelledby="model-chat-title">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><h2 id="model-chat-title" className="text-lg font-semibold">Chat with {name}</h2><p className="mt-1 max-w-3xl text-sm text-muted-foreground">Replies use {provider} and the completed turns below.{localWeights ? ' The first reply may take longer while the server loads the model.' : ''}</p></div>
      <div className="flex flex-wrap gap-2"><button type="button" className={BUTTON} onClick={exportTranscript} disabled={!exchanges.length}>Export transcript</button><button type="button" className={BUTTON} onClick={reset} disabled={busy || adjusting || !exchanges.length}>New conversation</button></div>
    </div>
    {disabledReason && <p role="alert" className="rounded border border-amber-400/40 p-3 text-sm">{disabledReason}</p>}
    <p className="text-xs text-muted-foreground">This conversation stays in this page. Export it to keep the replies and recorded generation evidence. Context starts from saved {promptRole} defaults. Current prompt source: {promptSource}. The role chooses defaults and presets without assigning an identity to the model.</p>
    {loadingDefaults && <p role="status" className="text-sm text-muted-foreground">Loading saved chat defaults…</p>}
    {restoringRevision && <p role="status" className="text-sm text-muted-foreground">Loading the saved model version…</p>}
    {defaultPromptIssue && <div role="alert" className="space-y-2 rounded border p-3 text-sm"><p>{defaultPromptIssue}</p><button type="button" className={BUTTON} onClick={() => updateSettings({ system_prompt: '' })}>Use no app system prompt</button></div>}
    {settings && <details className="rounded border p-3" open={!!defaultPromptIssue}><summary className="cursor-pointer text-sm font-medium">Prompt &amp; generation · ReACT {settings.enable_cot ? 'on' : 'off'}</summary>
      <p className="mt-2 text-xs text-muted-foreground">Changing these settings starts a fresh conversation. Export any replies you want to keep. Blank sampling fields use provider defaults.</p>
      <fieldset disabled={busy || adjusting || loadingDefaults || restoringRevision} className="mt-3 space-y-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block text-sm">Defaults / preset role<select value={promptRole} onChange={(event) => setPromptRole(event.target.value as GenerationRole)} className={INPUT}>{(['patient', 'doctor', 'evaluator'] as const).map((role) => <option key={role} value={role}>{role[0].toUpperCase() + role.slice(1)}</option>)}</select></label>
          <label className="block text-sm">System preset<select value={presetId} onChange={(event) => {
            const preset = presets.find((item) => String(item.id) === event.target.value)
            if (!preset) return
            updateSettings({ system_prompt: preset.prompt_text })
            setPresetId(String(preset.id)); setPromptSource(`${promptRole} preset: ${preset.name}`)
          }} className={INPUT}><option value="">Choose a preset</option>{presets.map((preset) => <option key={preset.id} value={String(preset.id)}>{preset.name}</option>)}</select></label>
        </div>
        <p className="text-xs text-muted-foreground">Prompt source: {promptSource}. Changing the role loads its saved prompt and generation choices without changing the selected model. An empty saved prompt means no app system message in this chat.</p>
        {presetIssue && <p role="status" className="text-xs text-muted-foreground">{presetIssue}</p>}
        <label className="block text-sm">Selected system prompt<textarea value={settings.system_prompt} maxLength={32000} rows={3} onChange={(event) => updateSettings({ system_prompt: event.target.value })} className={INPUT} placeholder="Optional instructions for this conversation" /></label>
        {!settings.system_prompt && <p className="text-xs text-muted-foreground">No app system prompt. The provider or chat template may add its own default.</p>}
        <label className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={settings.enable_cot} onChange={(event) => updateSettings({ enable_cot: event.target.checked })} /><span>Request ReACT reasoning<span className="block text-xs text-muted-foreground">Requests extra reasoning text before the answer and can change behavior. It does not expose internal computation or execute actions.</span></span></label>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <label className="text-sm">Temperature<input type="number" min="0" max="2" step="0.1" placeholder="Provider default" value={settings.temperature ?? ''} onChange={(event) => updateSettings({ temperature: numberValue(event.target.value) })} className={INPUT} /></label>
          <label className="text-sm">Top-p<input type="number" min="0.01" max="1" step="0.05" placeholder="Provider default" value={settings.top_p ?? ''} onChange={(event) => updateSettings({ top_p: numberValue(event.target.value) })} className={INPUT} /></label>
          <label className="text-sm">Maximum output tokens<input type="number" min="1" max="32768" placeholder="Provider default" value={settings.max_tokens ?? ''} onChange={(event) => updateSettings({ max_tokens: numberValue(event.target.value) })} className={INPUT} /></label>
          <label className="text-sm">Seed<input type="number" min="0" max="4294967295" placeholder="Random" value={settings.seed ?? ''} onChange={(event) => updateSettings({ seed: numberValue(event.target.value) })} className={INPUT} /></label>
        </div>
        {localWeights && <div className="grid grid-cols-2 gap-3">
          <label className="text-sm">Device<select value={settings.device} onChange={(event) => updateSettings({ device: event.target.value as ChatSettings['device'] })} className={INPUT}>{['auto', 'cuda', 'cpu', 'mps'].map((value) => <option key={value} value={value}>{value === 'auto' ? 'Server default' : value.toUpperCase()}</option>)}</select></label>
          <label className="text-sm">Precision<select value={settings.dtype} onChange={(event) => updateSettings({ dtype: event.target.value as ChatSettings['dtype'] })} className={INPUT}>{['bfloat16', 'float16', 'float32'].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        </div>}
        <button type="button" className={BUTTON} onClick={saveDefaults} disabled={!!defaultPromptIssue}>{`Save as ${promptRole} defaults`}</button>
        <p className="text-xs text-muted-foreground">Saves the system prompt, sampling and ReACT choices for new chats and tests. Model selection and the run seed are unchanged.</p>
      </fieldset>
      {savedMessage && <p role="status" className="mt-2 text-sm text-primary">{savedMessage}</p>}
    </details>}
    {actions && <div className="space-y-2 rounded border bg-muted/20 p-3">
      <div className="flex flex-wrap gap-2">
        <Link href={actions.test} className={BUTTON}>Run a test</Link>
        <Link href={actions.suite} className={BUTTON}>Compare test runs</Link>
        {actions.modify && <Link href={actions.modify} className={BUTTON}>Modify weights</Link>}
        {actions.compare && <Link href={actions.compare} className={BUTTON}>Compare weight versions</Link>}
      </div>
      <p className="text-xs text-muted-foreground">Tests and test comparisons open with this model, system prompt, sampling and ReACT choice, plus your latest message. Blank fields use the last reply’s recorded values when available; otherwise each workflow uses its own defaults. Set values explicitly for controlled comparisons. Review the form before starting.</p>
      {localWeights ? <p className="text-xs text-muted-foreground">Weight comparisons carry these generation settings. Weight modifications configure their own operation.</p> : <p className="text-xs text-muted-foreground">Weight modification needs a Transformers checkpoint on the analysis server. <Link href="/compare?view=find" className="text-primary underline">Find and download its open-weight version</Link> if one is available.</p>}
    </div>}
    {actionError && <p className="text-sm text-destructive">{actionError}</p>}
    <div role="log" aria-live="polite" aria-busy={busy} aria-label="Model conversation" className="max-h-[620px] space-y-5 overflow-y-auto rounded border bg-background p-3 sm:p-4">
      {!exchanges.length && !busy && <p className="py-8 text-center text-sm text-muted-foreground">Send a message to see how this model responds.</p>}
      {exchanges.map((turn, index) => {
        const reasoning = turn.response.reasoning || turn.response.metadata?.reasoning
        const source = turn.response.reasoning_source || turn.response.metadata?.reasoning_source
        const nativeReasoning = turn.response.metadata?.native_reasoning
        return <article key={`${index}-${turn.at}`} className="space-y-3">
          <div className="rounded bg-muted p-3"><p className="mb-1 text-xs font-semibold text-muted-foreground">You · turn {index + 1}</p><p className="whitespace-pre-wrap break-words text-sm">{turn.prompt}</p></div>
          <div className="space-y-2 border-l-2 border-primary pl-3"><p className="break-words text-xs font-semibold">{turn.response.provider} / {turn.response.model}</p>
            <TurnSystemPrompts prompts={turn.response.metadata?.request_system_prompts} />
            {typeof reasoning === 'string' && reasoning && <details className="rounded border border-dashed p-3 text-xs"><summary className="cursor-pointer font-medium">{source === 'react' ? 'Requested reasoning (ReACT)' : source === 'provider' || source === 'inline' ? 'Model-provided reasoning' : 'Recorded reasoning'}</summary><p className="mt-2 whitespace-pre-wrap break-words">{reasoning}</p></details>}
            {typeof nativeReasoning === 'string' && nativeReasoning && <details className="rounded border border-dashed p-3 text-xs"><summary className="cursor-pointer font-medium">Model-provided reasoning</summary><p className="mt-2 whitespace-pre-wrap break-words">{nativeReasoning}</p></details>}
            <p className="whitespace-pre-wrap break-words text-sm">{turn.response.content || '(The model returned no answer text.)'}</p>
            {!turn.response.content.trim() && <p className="text-xs text-muted-foreground">This exchange is omitted from later conversation context. Reasoning text is never used as the answer.</p>}
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
              {typeof turn.response.elapsed_seconds === 'number' && <span>{turn.response.elapsed_seconds.toFixed(2)} seconds</span>}
              {turn.response.finish_reason && <span>Finish: {turn.response.finish_reason}</span>}
              {turn.response.finish_reason === 'length' && <span className="text-amber-700 dark:text-amber-300">Output reached its token limit</span>}
            </div>
            {turn.response.generation?.notes?.map((note, index) => <p key={index} className="text-xs text-muted-foreground">{note}</p>)}
            <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">Generation evidence &amp; token usage</summary><pre className="mt-2 overflow-auto whitespace-pre-wrap break-words">{JSON.stringify({ provider: turn.response.provider, model: turn.response.model, generation: turn.response.generation, usage: turn.response.usage, system_prompt_source: turn.response.metadata?.request_system_prompts_source }, null, 2)}</pre></details>
          </div>
        </article>
      })}
      {busy && <div className="space-y-2"><p className="whitespace-pre-wrap break-words rounded bg-muted p-3 text-sm">{pendingPrompt}</p><p role="status" className="text-sm text-muted-foreground">Waiting for the model’s reply…</p></div>}
      <div ref={end} />
    </div>
    {settings && !loadingDefaults && !restoringRevision && !defaultPromptIssue && <ModelAdjustmentPanel
      key={JSON.stringify([provider, model, settings, exchanges.map((turn) => [turn.prompt, turn.response.content, turn.at])])}
      provider={provider} model={model} settings={chatSettingsForHandoff(settings, exchanges.at(-1)?.response)} exchanges={exchanges}
      parentId={activeRevisionId} disabled={busy || !!disabledReason} onUseVersion={useAdjustment} onBusyChange={setAdjusting}
    />}
    {error && <div role="alert" className="rounded border border-destructive/40 p-4 text-sm text-destructive">{error}<p className="mt-1">Your message is still here. Review the settings and retry.</p></div>}
    <form onSubmit={send} className="space-y-2">
      <label htmlFor="model-chat-message" className="text-sm font-medium">Message</label>
      <textarea id="model-chat-message" rows={3} maxLength={32000} value={draft} onChange={(event) => setDraft(event.target.value)} disabled={busy} className={INPUT} placeholder="Ask a question or continue the conversation" />
      <div className="flex items-center justify-between gap-3"><span className="text-xs text-muted-foreground">{exchanges.length} completed turn{exchanges.length === 1 ? '' : 's'} · up to 50 per conversation</span><button type="submit" disabled={busy || adjusting || !draft.trim() || loadingDefaults || restoringRevision || !!defaultPromptIssue || !!disabledReason} className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">{busy ? 'Waiting for reply…' : 'Send message'}</button></div>
    </form>
  </section>
}
