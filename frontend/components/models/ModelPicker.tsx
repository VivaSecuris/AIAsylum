import { useState } from 'react'
import Link from 'next/link'
import { DownloadModelButton } from '@/components/models/DownloadPanel'
import { ModelDiscovery, HuggingFaceAccess } from '@/components/models/ModelDiscovery'
import type { DiscoveredModel } from '@/lib/model-discovery'
import { availabilityLabel, canUseModel, useModelCatalog, useModelDownloads } from '@/lib/model-catalog'
import { belongsToExperiment, modelOrganizationKey, useModelOrganization } from '@/lib/model-organization'
import { modelChatUrl } from '@/lib/model-chat'
import { formatApiError } from '@/lib/utils'

interface ModelPickerProps {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  description?: string
  disabled?: boolean
}

const INPUT = 'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

export function ModelPicker({ id, label, value, onChange, description, disabled }: ModelPickerProps) {
  const { data, isLoading, error, refetch } = useModelCatalog()
  const [manual, setManual] = useState(false)
  const [discover, setDiscover] = useState(false)
  const [search, setSearch] = useState('')
  const [experiment, setExperiment] = useState('')
  const [discovered, setDiscovered] = useState<DiscoveredModel | null>(null)
  const { data: organization } = useModelOrganization()
  const models = data?.models ?? []
  const selected = models.find((model) => model.model_ref === value)
  const chosenDiscovery = discovered?.id === value ? discovered : null
  const gated = selected?.gated || chosenDiscovery?.gated
  const metadata = (ref: string) => organization?.items[modelOrganizationKey(ref)]
  const visible = models.filter((model) => model.availability !== 'deleted' && belongsToExperiment(metadata(model.model_ref), experiment) && [model.name, model.model_ref, metadata(model.model_ref)?.label, metadata(model.model_ref)?.notes].some((part) => part?.toLowerCase().includes(search.toLowerCase().trim())))
  const { data: downloadData } = useModelDownloads()
  const download = downloadData?.downloads.find((d) => d.repo_id === value)
  const manualEntry = manual || (!!value && !!data && !selected)

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <label htmlFor={id} className="text-sm font-medium">{label}</label>
        <button type="button" disabled={disabled} onClick={() => {
          if (manualEntry && !selected) onChange('')
          setManual(!manualEntry)
        }} className="text-xs text-primary hover:underline disabled:opacity-50">
          {manualEntry ? 'Choose from library' : 'Enter an ID or path'}
        </button>
      </div>
      {manualEntry ? (
        <>
          <input id={id} value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)} placeholder="Hugging Face ID or checkpoint path on the server" className={INPUT} />
          <p className="mt-1 text-xs text-muted-foreground">Paths must exist on the connected analysis server. Model fit is checked before analysis.</p>
        </>
      ) : (<>
        <div className="flex flex-wrap gap-2">
          <input aria-label={`Search ${label.toLowerCase()} library`} value={search} onChange={(event) => setSearch(event.target.value)} disabled={disabled} placeholder="Filter by model name, family or notes…" className={`${INPUT} min-w-[160px] flex-1`} />
          {!!organization?.experiments.length && <select aria-label={`${label} experiment filter`} value={experiment} onChange={(event) => setExperiment(event.target.value)} className={`${INPUT} w-auto max-w-full`}><option value="">All experiments</option>{organization.experiments.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select>}
        </div>
        <select id={id} value={value} disabled={disabled || isLoading} onChange={(event) => onChange(event.target.value)} className={INPUT}>
          <option value="">{isLoading ? 'Loading model library…' : 'Choose a model'}</option>
          {!!value && !visible.some((model) => model.model_ref === value) && <option value={value}>{metadata(value)?.label || selected?.name || value} · Current selection</option>}
          {(['custom', 'ready', 'discover'] as const).map((kind) => {
            const group = visible.filter((model) => kind === 'custom' ? model.kind === 'custom' : model.kind === 'base' && (kind === 'ready' ? model.availability === 'ready' : model.availability !== 'ready'))
            return group.length > 0 && (
              <optgroup key={kind} label={kind === 'custom' ? 'Your custom models' : kind === 'ready' ? 'Base models ready on server' : 'Common models to download'}>
                {group.map((model) => <option key={model.id} value={model.model_ref} disabled={!canUseModel(model)}>{metadata(model.model_ref)?.label || model.name} · {availabilityLabel(model)}{model.gated ? ' · Access required' : ''}</option>)}
              </optgroup>
            )
          })}
        </select>
        {!!search && visible.length === 0 && <p className="text-xs text-muted-foreground">No library matches. Use “Find common models” to search Hugging Face.</p>}
      </>)}
      <button type="button" onClick={() => setDiscover((open) => !open)} disabled={disabled} aria-expanded={discover} aria-controls={`${id}-discover`} className="rounded-md border px-3 py-1.5 text-xs font-medium text-primary hover:bg-muted disabled:opacity-50">{discover ? 'Close model search' : 'Find common models'}</button>
      {discover && <div id={`${id}-discover`}><ModelDiscovery selectedId={value} disabled={disabled} onSelect={(model) => { setDiscovered(model); onChange(model.id); setManual(false); setSearch(''); setExperiment(''); setDiscover(false) }} /></div>}
      {!!value && <Link href={modelChatUrl(selected?.provider || 'transformers', value)} className="inline-block text-xs text-primary hover:underline">Chat &amp; actions for this model →</Link>}
      {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
      {((selected?.kind === 'base' && selected.availability === 'download_required') || (!selected && chosenDiscovery) || download?.active) && (
        <div className="mt-2 space-y-1">
          <DownloadModelButton repoId={value} download={download} className="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-50" />
          <p className="text-xs text-muted-foreground">Fetch the weights now rather than at the start of the run. Progress is on the <Link href="/compare?view=models" className="text-primary hover:underline">Models</Link> page.</p>
        </div>
      )}
      {!!gated && <HuggingFaceAccess gated={gated} modelId={value} />}
      {selected?.reason && <p className={`mt-1 text-xs ${canUseModel(selected) ? 'text-muted-foreground' : 'text-destructive'}`}>{selected.reason}</p>}
      {error && <div className="mt-2 text-xs text-destructive" role="alert">{formatApiError(error, 'Could not load the model library.')} <button type="button" onClick={() => refetch()} className="underline">Retry</button></div>}
      {!isLoading && !error && data && models.length === 0 && <p className="mt-1 text-xs text-muted-foreground">No models found on this server. Enter a Hugging Face ID or an existing server path.</p>}
    </div>
  )
}
