import { useEffect, useRef, useState } from 'react'
import { apiClient } from '@/lib/api'
import { buildChatRequest, chatErrorMessage, type ChatExchange, type ChatSettings } from '@/lib/model-chat'

const INPUT = 'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm disabled:opacity-50'
const BUTTON = 'rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50'

export function ModelChat({ name }: { name: string }) {
  const [exchanges, setExchanges] = useState<ChatExchange[]>([])
  const [draft, setDraft] = useState('')
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [settings, setSettings] = useState<ChatSettings>({ system_prompt: '', temperature: 0, max_tokens: 256, device: 'auto', dtype: 'bfloat16' })
  const requestVersion = useRef(0)
  const end = useRef<HTMLDivElement>(null)
  const busy = pendingPrompt !== null

  useEffect(() => {
    setExchanges([])
    setDraft('')
    setError(null)
    setPendingPrompt(null)
    requestVersion.current += 1
    return () => { requestVersion.current += 1 }
  }, [name])
  useEffect(() => { end.current?.scrollIntoView({ block: 'nearest' }) }, [exchanges.length, busy])

  function updateSettings(patch: Partial<ChatSettings>) {
    requestVersion.current += 1
    setSettings((previous) => ({ ...previous, ...patch }))
    setExchanges([])
    setError(null)
  }

  function reset() {
    requestVersion.current += 1
    setExchanges([])
    setDraft('')
    setError(null)
  }

  function exportTranscript() {
    const payload = { model_name: name, exported_at: new Date().toISOString(), settings, exchanges }
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${name}-chat-${new Date().toISOString().slice(0, 10)}.json`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  async function send(event: React.FormEvent) {
    event.preventDefault()
    if (busy) return
    setError(null)
    const prompt = draft.trim()
    let request
    try { request = buildChatRequest(exchanges, prompt, settings) } catch (cause) { setError(chatErrorMessage(cause)); return }
    const version = ++requestVersion.current
    setPendingPrompt(prompt)
    try {
      const response = await apiClient.chatWithModel(name, request)
      if (requestVersion.current !== version) return
      setExchanges((previous) => [...previous, { prompt, response, at: new Date().toISOString() }])
      setDraft('')
    } catch (cause) {
      if (requestVersion.current === version) setError(chatErrorMessage(cause))
    } finally {
      if (requestVersion.current === version) setPendingPrompt(null)
    }
  }

  return <section className="space-y-4 rounded-lg border bg-card p-4 sm:p-6" aria-labelledby="checkpoint-chat-title">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><h2 id="checkpoint-chat-title" className="text-lg font-semibold">Chat with this checkpoint</h2><p className="mt-1 max-w-3xl text-sm text-muted-foreground">Each reply uses this checkpoint and all completed turns below. The first reply may take several minutes while weights load or another GPU job finishes.</p></div>
      <div className="flex gap-2"><button type="button" className={BUTTON} onClick={exportTranscript} disabled={!exchanges.length}>Export transcript</button><button type="button" className={BUTTON} onClick={reset} disabled={busy || !exchanges.length}>New conversation</button></div>
    </div>
    <p className="text-xs text-muted-foreground">This conversation stays in this page; export it to keep the replies, provenance and measurements. Live probe scores and simultaneous original-model replies are unavailable here. Use the model&apos;s evaluation links for recorded comparisons.</p>
    <details className="rounded border p-3"><summary className="cursor-pointer text-sm font-medium">Generation settings</summary>
      <p className="mt-2 text-xs text-muted-foreground">Changing settings starts a fresh conversation. Export the current transcript first if you need to keep it.</p>
      <fieldset disabled={busy} className="mt-3 space-y-3">
        <label className="block text-sm">System prompt<textarea value={settings.system_prompt} maxLength={32000} rows={2} onChange={(event) => updateSettings({ system_prompt: event.target.value })} className={INPUT} placeholder="Optional instructions for this conversation" /></label>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <label className="text-sm">Temperature<input type="number" min="0" max="2" step="0.1" value={settings.temperature} onChange={(event) => updateSettings({ temperature: Number(event.target.value) })} className={INPUT} /></label>
          <label className="text-sm">Maximum output tokens<input type="number" min="1" max="4096" value={settings.max_tokens} onChange={(event) => updateSettings({ max_tokens: Number(event.target.value) })} className={INPUT} /></label>
          <label className="text-sm">Device<select value={settings.device} onChange={(event) => updateSettings({ device: event.target.value as ChatSettings['device'] })} className={INPUT}>{['auto', 'cuda', 'cpu', 'mps'].map((value) => <option key={value} value={value}>{value === 'auto' ? 'Server default' : value.toUpperCase()}</option>)}</select></label>
          <label className="text-sm">Precision<select value={settings.dtype} onChange={(event) => updateSettings({ dtype: event.target.value as ChatSettings['dtype'] })} className={INPUT}>{['bfloat16', 'float16', 'float32'].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        </div>
      </fieldset>
    </details>
    <div role="log" aria-live="polite" aria-busy={busy} aria-label="Checkpoint conversation" className="max-h-[620px] space-y-5 overflow-y-auto rounded border bg-background p-3 sm:p-4">
      {!exchanges.length && !busy && <p className="py-8 text-center text-sm text-muted-foreground">Send a question to inspect how this checkpoint responds.</p>}
      {exchanges.map((turn, index) => <article key={`${index}-${turn.at}`} className="space-y-3">
        <div className="rounded bg-muted p-3"><p className="mb-1 text-xs font-semibold text-muted-foreground">You · turn {index + 1}</p><p className="whitespace-pre-wrap break-words text-sm">{turn.prompt}</p></div>
        <div className="space-y-2 border-l-2 border-primary pl-3"><p className="text-xs font-semibold">{name}</p><p className="whitespace-pre-wrap break-words text-sm">{turn.response.content || '(The model returned no text.)'}</p>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            {typeof turn.response.elapsed_seconds === 'number' && <span>{turn.response.elapsed_seconds.toFixed(2)} seconds</span>}
            {typeof turn.response.refused === 'boolean' && <span>{turn.response.refused ? 'Refusal phrase detected' : 'No refusal phrase detected'}</span>}
            {turn.response.finish_reason && <span>Finish: {turn.response.finish_reason}</span>}
            {turn.response.truncated && <span className="text-amber-700 dark:text-amber-300">Output reached its token limit</span>}
          </div>
          {turn.response.usage && Object.keys(turn.response.usage).length > 0 && <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">Token usage</summary><dl className="mt-1 grid grid-cols-2 gap-x-3">{Object.entries(turn.response.usage).map(([key, value]) => <div key={key}><dt className="inline">{key.replaceAll('_', ' ')}: </dt><dd className="inline">{value}</dd></div>)}</dl></details>}
        </div>
      </article>)}
      {busy && <div className="space-y-2"><p className="whitespace-pre-wrap break-words rounded bg-muted p-3 text-sm">{pendingPrompt}</p><p role="status" className="text-sm text-muted-foreground">Waiting for the checkpoint&apos;s reply…</p></div>}
      <div ref={end} />
    </div>
    <p className="text-xs text-muted-foreground">Refusal detection matches phrases in the visible answer. It can miss refusals or flag quotations, and does not measure whether a reply is safe or correct.</p>
    {error && <div role="alert" className="rounded border border-destructive/40 p-4 text-sm text-destructive">{error}<p className="mt-1">No reply was added. Edit the message or settings and retry.</p></div>}
    <form onSubmit={send} className="space-y-2">
      <label htmlFor="checkpoint-chat-message" className="text-sm font-medium">Message</label>
      <textarea id="checkpoint-chat-message" rows={3} maxLength={32000} value={draft} onChange={(event) => setDraft(event.target.value)} disabled={busy} className={INPUT} placeholder="Ask a question or continue the conversation" />
      <div className="flex items-center justify-between gap-3"><span className="text-xs text-muted-foreground">{exchanges.length} completed turn{exchanges.length === 1 ? '' : 's'} · up to 50 per conversation</span><button type="submit" disabled={busy || !draft.trim()} className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">{busy ? 'Waiting for reply…' : 'Send message'}</button></div>
    </form>
  </section>
}
