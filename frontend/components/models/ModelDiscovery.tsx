import { useId, useState } from 'react'
import { Search } from 'lucide-react'
import { useHuggingFaceStatus, useModelDiscovery, type DiscoveredModel } from '@/lib/model-discovery'
import { availabilityLabel, useModelCatalog } from '@/lib/model-catalog'
import { formatApiError } from '@/lib/utils'

const BUTTON = 'rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

export function HuggingFaceAccess({ gated, modelId }: { gated?: boolean | string; modelId?: string }) {
  const { data, error, refetch, isFetching } = useHuggingFaceStatus()
  return <div className="space-y-2 text-xs text-muted-foreground">
    <p>{data ? data.token_configured ? 'Hugging Face token configured on the server. Access still depends on the model’s permissions.' : 'Public models need no Hugging Face sign-in. No token is configured on this server.' : error ? 'Could not check Hugging Face access on the server.' : 'Checking Hugging Face access…'}</p>
    {!!gated && <p className="rounded-md bg-amber-50 p-2 text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">This model requires Hugging Face access. {modelId && <a href={`https://huggingface.co/${modelId}`} target="_blank" rel="noreferrer" className="underline">Open its model page</a>} to review the license and request access, then configure a read token on the analysis server.</p>}
    <details>
      <summary className="cursor-pointer text-primary">Using a gated or private model</summary>
      <div className="mt-2 space-y-2 rounded-md border p-3">
        <p>Sign into Hugging Face and obtain access on the model’s page. Create a <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="text-primary underline">read token</a> with permission for that repository.</p>
        <p>On the analysis server, activate its Python environment and run <code className="font-mono">hf auth login</code> as the account running the API. Enter the token at that private terminal prompt. Then restart the API when no jobs are running and refresh the status here.</p>
        <p>Signing into the website does not sign the remote GPU server in. Keep tokens out of model IDs and experiment notes.</p>
        <button type="button" className="text-primary underline disabled:opacity-50" disabled={isFetching} onClick={() => refetch()}>Refresh server access status</button>
      </div>
    </details>
  </div>
}

/** Public Hub search is separate from local search so checkpoint paths stay local. */
export function ModelDiscovery({ onSelect, selectedId, disabled = false }: { onSelect: (model: DiscoveredModel) => void; selectedId?: string; disabled?: boolean }) {
  const id = useId()
  const [draft, setDraft] = useState('')
  const [query, setQuery] = useState('')
  const { data, isFetching, error, refetch } = useModelDiscovery(query)
  const { data: catalog } = useModelCatalog()
  function search() {
    const next = draft.trim()
    if (next === query) refetch()
    else setQuery(next)
  }
  return <section className="space-y-3 rounded-lg border bg-muted/10 p-3" aria-label="Find Hugging Face models">
    <div>
      <label htmlFor={id} className="text-sm font-medium">Find common models or search Hugging Face</label>
      <div className="mt-1 flex gap-2">
        <input id={id} value={draft} maxLength={80} disabled={disabled} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); search() } }} placeholder="Try Llama, Gemma, Mistral, Qwen…" className="min-w-0 flex-1 rounded-md border bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" />
        <button type="button" onClick={search} disabled={disabled || isFetching} className={`${BUTTON} inline-flex items-center gap-1.5`}><Search className="h-4 w-4" />Search</button>
      </div>
    </div>
    <p className="text-xs text-muted-foreground">{query ? `Results for “${query}”` : 'Common model families and sizes'} · Selecting a model does not download it. Resource and architecture support must still be checked before analysis.</p>
    {isFetching && <p role="status" className="text-xs text-muted-foreground">Loading models…</p>}
    {error && <p role="alert" className="text-xs text-destructive">{formatApiError(error, 'Could not search Hugging Face.')} <button type="button" onClick={() => refetch()} className="underline">Retry</button></p>}
    {data?.warning && <p role="status" className="text-xs text-amber-700 dark:text-amber-300">{data.warning}</p>}
    {data && <ul className="max-h-72 divide-y overflow-y-auto rounded-md border bg-background" aria-label="Hugging Face model results">
      {data.models.map((model) => {
        const local = catalog?.models.find((entry) => entry.model_ref === model.id)
        return <li key={model.id}>
          <button type="button" disabled={disabled} onClick={() => onSelect(model)} aria-pressed={selectedId === model.id} className={`w-full px-3 py-2.5 text-left hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring ${selectedId === model.id ? 'bg-primary/10' : ''}`}>
            <span className="block break-all text-sm font-medium">{model.id}</span>
            <span className="mt-1 block text-xs text-muted-foreground">{model.family} · {local?.availability === 'ready' ? availabilityLabel(local) : 'Download required'} · {model.gated ? 'Access required' : 'Public'}{selectedId === model.id ? ' · Selected' : ''}</span>
          </button>
        </li>
      })}
      {data.models.length === 0 && <li className="p-3 text-sm text-muted-foreground">No matching text generation models. Try a family name or enter an exact repository ID in the model picker.</li>}
    </ul>}
    {!!query && <button type="button" onClick={() => { setDraft(''); setQuery('') }} className="text-xs text-primary hover:underline">Show common models</button>}
    <HuggingFaceAccess />
  </section>
}
