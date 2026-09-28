import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { ArrowRight, BarChart3, Brain, CheckCircle2, Download, GitBranch, GitCompare, RefreshCw, Search } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { LineageGraph } from '@/components/models/LineageGraph'
import { ModelVisualizations } from '@/components/models/ModelVisualizations'
import { HuggingFaceAccess, ModelDiscovery } from '@/components/models/ModelDiscovery'
import type { DiscoveredModel } from '@/lib/model-discovery'
import { ExperimentToolbar } from '@/components/models/ExperimentToolbar'
import { OrganizationEditor } from '@/components/models/OrganizationEditor'
import { DeleteCustomButton } from '@/components/models/DeleteCustomButton'
import { belongsToExperiment, modelOrganizationKey, useModelOrganization, type ModelOrganization } from '@/lib/model-organization'
import { DownloadModelButton, DownloadPanel, RemoveCachedButton } from '@/components/models/DownloadPanel'
import { availabilityLabel, canUseModel, editDescription, isCachedBaseModel, useModelCatalog, useModelDownloads, type ModelCatalogEntry, type ModelDownload } from '@/lib/model-catalog'
import { modelChatUrl } from '@/lib/model-chat'
import { formatApiError } from '@/lib/utils'

const ACTION = 'inline-flex items-center justify-center gap-2 rounded-md border px-3 py-2 text-sm font-medium transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

function modelUrl(pathname: string, query: Record<string, string>) {
  return { pathname, query }
}

function readableSize(bytes: number | null) {
  if (bytes == null || bytes <= 0) return null
  const gib = bytes / 1024 ** 3
  return gib >= 1 ? `${gib.toFixed(1)} GiB` : `${(bytes / 1024 ** 2).toFixed(0)} MiB`
}

function ModelCard({ model, allModels, download, organization, onViewHistory }: { model: ModelCatalogEntry; allModels: ModelCatalogEntry[]; download?: ModelDownload; organization?: ModelOrganization; onViewHistory: (model: ModelCatalogEntry) => void }) {
  const item = organization?.items[modelOrganizationKey(model.model_ref)]
  const displayName = item?.label || model.name
  const experiments = organization?.experiments.filter((experiment) => item?.experiment_ids.includes(experiment.id)) ?? []
  const usable = canUseModel(model)
  const custom = model.kind === 'custom'
  const original = allModels.find((entry) => entry.model_ref === model.source_model)
  const originalName = original?.name ?? model.source_model?.split('/').filter(Boolean).pop()
  const compareEnabled = usable && !!model.source_model && (!original || canUseModel(original))
  const history = model.history ?? { test_runs: 0, interp_runs: 0, weight_runs: 0 }
  const size = readableSize(model.size_bytes)
  const ready = model.availability === 'ready'

  return (
    <article className="flex h-full flex-col rounded-xl border bg-card shadow-sm">
      <div className="space-y-4 p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <span className={custom ? 'rounded-full bg-primary/10 px-2.5 py-1 font-medium text-primary' : 'rounded-full bg-muted px-2.5 py-1 text-muted-foreground'}>{custom ? 'Custom model' : 'Base model'}</span>
          <span className={`inline-flex items-center gap-1.5 ${ready ? 'text-emerald-700 dark:text-emerald-400' : usable ? 'text-muted-foreground' : 'text-amber-700 dark:text-amber-400'}`}>
            {ready ? <CheckCircle2 className="h-3.5 w-3.5" /> : model.availability === 'download_required' ? <Download className="h-3.5 w-3.5" /> : null}
            {availabilityLabel(model)}
          </span>
        </div>
        <div>
          <h2 className="break-words text-lg font-semibold"><Link href={modelChatUrl(model.provider, model.model_ref)} className="hover:text-primary hover:underline">{displayName}</Link></h2>
          {item?.label && <p className="mt-1 break-all text-xs text-muted-foreground">{model.name}</p>}
          {item?.notes && <p className="mt-2 whitespace-pre-wrap break-words text-sm">{item.notes}</p>}
          {experiments.length > 0 && <div className="mt-2 flex flex-wrap gap-1">{experiments.map((experiment) => <span key={experiment.id} className="rounded-full bg-muted px-2 py-1 text-xs">{experiment.name}</span>)}</div>}
          <p className="mt-1 text-sm text-muted-foreground">{custom ? editDescription(model) : 'Original pretrained weights'}</p>
          {custom && <p className="mt-2 text-sm"><span className="text-muted-foreground">Original: </span>{originalName || 'Not recorded'}</p>}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          {size && <span>{size} on disk</span>}
          <span>{history.test_runs > 0 ? `${history.test_runs} test run${history.test_runs === 1 ? '' : 's'}` : 'No test runs recorded'}</span>
          {history.weight_runs > 0 && <span>{history.weight_runs} weight experiment{history.weight_runs === 1 ? '' : 's'}</span>}
          <span>{history.interp_runs > 0 ? `${history.interp_runs} analysis run${history.interp_runs === 1 ? '' : 's'}` : 'No analyses recorded'}</span>
        </div>
        {!usable && <p className="rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">{model.reason || 'This checkpoint is unavailable on the connected server. Restore its files before starting a run.'}</p>}
        {custom && usable && !model.source_model && <p className="text-xs text-muted-foreground">The original model is not recorded. Choose both models in Interpretability to prepare a comparison.</p>}
        {!custom && model.availability === 'download_required' && <DownloadModelButton repoId={model.model_ref} download={download} className={`${ACTION} w-full`} />}
        {!custom && download?.active && model.availability !== 'download_required' && <DownloadModelButton repoId={model.model_ref} download={download} />}
        {isCachedBaseModel(model) && !download?.active && <RemoveCachedButton repoId={model.model_ref} sizeBytes={model.size_bytes} />}
        {custom && model.manifest && ['ready', 'incomplete'].includes(model.availability) && <DeleteCustomButton names={[model.name]} />}
      </div>
      <div className="mt-auto space-y-3 border-t p-4">
        {custom && (
          compareEnabled ? <Link href={modelUrl('/interp', { mode: 'model_diff', model_a: model.source_model!, model_b: model.model_ref })} className={`${ACTION} w-full border-primary bg-primary text-primary-foreground hover:bg-primary/90`}>
            <GitCompare className="h-4 w-4" /> Compare with original
          </Link> : <button disabled className={`${ACTION} w-full cursor-not-allowed opacity-50`} title={!model.source_model ? 'Original model is not recorded' : 'Both checkpoints must be available'}><GitCompare className="h-4 w-4" /> Compare with original</button>
        )}
        {custom && <p className="text-center text-xs text-muted-foreground">Compare layer activations on one prompt.</p>}
        {custom && compareEnabled && <Link href={modelUrl('/weights', { kind: 'compare', source_model: model.source_model!, modified_model: model.model_ref })} className="block text-center text-xs text-primary hover:underline">Compare behavior on the same prompts →</Link>}
        <Link href={modelChatUrl(model.provider, model.model_ref)} className={`${ACTION} w-full`}>Chat &amp; model actions</Link>
        <div className="flex gap-2">
          {usable ? <>
            <Link href={modelUrl('/interp', { mode: 'single', model_a: model.model_ref })} className={`${ACTION} flex-1`}><Brain className="h-4 w-4" /> Analyze</Link>
            <Link href={modelUrl('/create-test', { provider: model.provider, model: model.model_ref })} className={`${ACTION} flex-1`}><BarChart3 className="h-4 w-4" /> Evaluate</Link>
          </> : <>
            <button disabled className={`${ACTION} flex-1 cursor-not-allowed opacity-50`}><Brain className="h-4 w-4" /> Analyze</button>
            <button disabled className={`${ACTION} flex-1 cursor-not-allowed opacity-50`}><BarChart3 className="h-4 w-4" /> Evaluate</button>
          </>}
        </div>
        <button type="button" onClick={() => onViewHistory(model)} className={`${ACTION} w-full`}><GitBranch className="h-4 w-4" /> View history</button>
        <OrganizationEditor itemKey={modelOrganizationKey(model.model_ref)} originalLabel={model.name} />
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer hover:text-foreground">Model details</summary>
          <dl className="mt-2 space-y-2 break-all">
            <div><dt className="font-medium">Model reference</dt><dd className="font-mono">{model.model_ref}</dd></div>
            {model.source_model && <div><dt className="font-medium">Original reference</dt><dd className="font-mono">{model.source_model}</dd></div>}
            {model.created_at && <div><dt className="font-medium">Saved</dt><dd>{new Date(model.created_at).toLocaleString()}</dd></div>}
            {model.reason && usable && <div><dt className="font-medium">Availability</dt><dd>{model.reason}</dd></div>}
          </dl>
          {model.run_id != null && <Link href={`/weights/${model.run_id}`} className="mt-2 inline-block text-primary hover:underline">View the editing run →</Link>}
        </details>
      </div>
    </article>
  )
}

export default function ModelsPage() {
  const router = useRouter()
  const { data, isLoading, error, refetch, isFetching } = useModelCatalog()
  const { data: downloadData } = useModelDownloads()
  const { data: organization } = useModelOrganization()
  const [experimentId, setExperimentId] = useState('')
  const downloads = downloadData?.downloads ?? []
  const [search, setSearch] = useState('')
  const [view, setView] = useState<'charts' | 'models' | 'history' | 'find'>('charts')
  const [discovered, setDiscovered] = useState<DiscoveredModel | null>(null)
  const [focusModel, setFocusModel] = useState<string | undefined>()
  const [kind, setKind] = useState<'all' | 'custom' | 'base'>('all')
  const [showDeleted, setShowDeleted] = useState(false)
  const models = data?.models ?? []
  const customCount = models.filter((model) => model.kind === 'custom' && model.availability !== 'deleted').length
  const deletable = models.filter((model) => model.kind === 'custom' && model.manifest && ['ready', 'incomplete'].includes(model.availability)).map((model) => model.name)
  const readyCount = models.filter((model) => model.availability === 'ready').length
  useEffect(() => {
    if (!router.isReady) return
    setView(['charts', 'models', 'history', 'find'].includes(String(router.query.view)) ? router.query.view as 'charts' | 'models' | 'history' | 'find' : typeof router.query.search === 'string' ? 'models' : 'charts')
    setFocusModel(typeof router.query.model === 'string' ? router.query.model : undefined)
    setExperimentId(typeof router.query.experiment === 'string' ? router.query.experiment : '')
  }, [router.isReady, router.query.view, router.query.model, router.query.experiment, router.query.search])
  useEffect(() => {
    if (router.isReady) setSearch(typeof router.query.search === 'string' ? router.query.search : '')
  }, [router.isReady, router.query.search])

  // Only explicit workspace actions write the URL. Hydration and history
  // navigation read it without triggering another replacement.
  function updateWorkspace(next: { view?: 'charts' | 'models' | 'history' | 'find'; model?: string; experiment?: string; clearSearch?: boolean }) {
    const nextView = next.view ?? view
    const nextModel = next.model ?? focusModel ?? ''
    const nextExperiment = next.experiment ?? experimentId
    setView(nextView)
    setFocusModel(nextModel || undefined)
    setExperimentId(nextExperiment)
    const query = { ...router.query }
    query.view = nextView
    if (nextModel) query.model = nextModel
    else delete query.model
    if (nextExperiment) query.experiment = nextExperiment
    else delete query.experiment
    if (next.clearSearch) { delete query.search; setSearch('') }
    void router.replace({ pathname: router.pathname, query }, undefined, { shallow: true, scroll: false })
  }

  function viewHistory(model: ModelCatalogEntry) {
    updateWorkspace({ view: 'history', model: model.model_ref })
  }
  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return models.filter((model) => {
      if (!showDeleted && model.availability === 'deleted') return false
      const item = organization?.items[modelOrganizationKey(model.model_ref)]
      const experimentNames = organization?.experiments.filter((entry) => item?.experiment_ids.includes(entry.id)).map((entry) => entry.name) ?? []
      return (kind === 'all' || model.kind === kind) && belongsToExperiment(item, experimentId) && (!needle || [model.name, model.model_ref, model.source_model, editDescription(model), item?.label, item?.notes, ...experimentNames].some((value) => value?.toLowerCase().includes(needle)))
    })
      .sort((a, b) => Number(b.kind === 'custom') - Number(a.kind === 'custom') || a.name.localeCompare(b.name, undefined, { numeric: true }))
  }, [models, search, kind, organization, experimentId, showDeleted])

  return (
    <Layout>
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <Link href="/models" className="mb-2 inline-block text-sm text-primary hover:underline">← All models</Link>
            <h1 className="text-3xl font-bold">Comparisons and model history</h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">Explore model visualizations, compare experiments, and organize checkpoints and their history.</p>
          </div>
          {data?.location && <p className="rounded-md border px-3 py-2 text-xs text-muted-foreground">Connected server: <span className="font-medium text-foreground">{data.location}</span></p>}
        </div>

        <div className="flex flex-wrap gap-1 border-b" role="tablist" aria-label="Model workspace">
          <button id="charts-tab" role="tab" aria-selected={view === 'charts'} aria-controls="charts-panel" onClick={() => updateWorkspace({ view: 'charts' })} className={`inline-flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium ${view === 'charts' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-foreground'}`}><BarChart3 className="h-4 w-4" />Charts &amp; visualizations</button>
          <button id="models-tab" role="tab" aria-selected={view === 'models'} aria-controls="models-panel" onClick={() => updateWorkspace({ view: 'models' })} className={`border-b-2 px-4 py-3 text-sm font-medium ${view === 'models' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-foreground'}`}>Models</button>
          <button id="history-tab" role="tab" aria-selected={view === 'history'} aria-controls="history-panel" onClick={() => updateWorkspace({ view: 'history' })} className={`inline-flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium ${view === 'history' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-foreground'}`}><GitBranch className="h-4 w-4" />History graph</button>
          <button id="find-models-tab" role="tab" aria-selected={view === 'find'} aria-controls="find-models-panel" onClick={() => updateWorkspace({ view: 'find' })} className={`inline-flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium ${view === 'find' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-foreground'}`}><Search className="h-4 w-4" />Find models</button>
        </div>
        {(view === 'models' || view === 'history') && <ExperimentToolbar value={experimentId} onChange={(id) => updateWorkspace({ experiment: id })} />}
        {view === 'charts' ? <div id="charts-panel" role="tabpanel" aria-labelledby="charts-tab"><ModelVisualizations /></div> : view === 'find' ? <div id="find-models-panel" role="tabpanel" aria-labelledby="find-models-tab" className="space-y-4">
          <ModelDiscovery onSelect={setDiscovered} selectedId={discovered?.id} />
          {discovered && <section className="space-y-3 rounded-xl border bg-card p-5" aria-label="Selected discovered model">
            <div><h2 className="break-all text-lg font-semibold"><Link href={modelChatUrl('transformers', discovered.id)} className="hover:text-primary hover:underline">{discovered.id}</Link></h2><p className="mt-1 text-sm text-muted-foreground">{discovered.family} · {models.some((model) => model.model_ref === discovered.id && model.availability === 'ready') ? 'Ready on server' : 'Download required on the connected server'}</p></div>
            {discovered.gated && <HuggingFaceAccess gated={discovered.gated} modelId={discovered.id} />}
            <div className="flex flex-wrap gap-2"><Link href={modelChatUrl('transformers', discovered.id)} className={ACTION}>Chat &amp; model actions</Link>{!models.some((model) => model.model_ref === discovered.id && model.availability === 'ready') && <DownloadModelButton repoId={discovered.id} download={downloads.find((download) => download.repo_id === discovered.id)} className={ACTION} />}<Link href={modelUrl('/interp', { mode: 'single', model_a: discovered.id })} className={ACTION}><Brain className="h-4 w-4" /> Prepare analysis</Link><Link href={modelUrl('/create-test', { provider: 'transformers', model: discovered.id })} className={ACTION}><BarChart3 className="h-4 w-4" /> Prepare evaluation</Link></div>
            <p className="text-xs text-muted-foreground">Review settings and resource checks before starting a run. Downloaded models appear in the Models tab.</p>
            <OrganizationEditor itemKey={modelOrganizationKey(discovered.id)} originalLabel={discovered.name} />
          </section>}
          <DownloadPanel location={data?.location} />
        </div> : view === 'history' ? <div id="history-panel" role="tabpanel" aria-labelledby="history-tab"><LineageGraph focusModel={focusModel} onFocusModelChange={(model) => updateWorkspace({ model: model || '' })} experimentId={experimentId} /></div> : <div id="models-panel" role="tabpanel" aria-labelledby="models-tab" className="space-y-6">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-1 rounded-lg bg-muted p-1" role="group" aria-label="Filter models">
              {([['all', 'All models'], ['custom', 'Custom models'], ['base', 'Base models']] as const).map(([value, label]) => <button type="button" key={value} aria-pressed={kind === value} onClick={() => setKind(value)} className={`rounded-md px-3 py-2 text-sm font-medium ${kind === value ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>{label}{value === 'custom' && data ? ` (${customCount})` : ''}</button>)}
            </div>
            <div className="flex flex-1 items-center gap-2 sm:max-w-sm">
              <div className="relative flex-1"><Search aria-hidden="true" className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" /><input aria-label="Search models" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search models, names or notes…" className="w-full rounded-md border bg-background py-2 pl-9 pr-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" /></div>
              <button type="button" aria-label="Refresh model library" title="Refresh model library" onClick={() => refetch()} disabled={isFetching} className={`${ACTION} px-2.5 disabled:opacity-50`}><RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} /></button>
            </div>
          </div>
          {data && <p className="text-xs text-muted-foreground">{models.length} models · {readyCount} ready on the connected server{data.location ? ` · ${data.location}` : ''}. Saved checkpoints appear before they have evaluation results.</p>}
          <div className="flex flex-wrap items-start justify-between gap-3"><label className="flex items-center gap-2 text-xs text-muted-foreground"><input type="checkbox" checked={showDeleted} onChange={(event) => setShowDeleted(event.target.checked)} />Show deleted checkpoints and their history</label><DeleteCustomButton names={deletable} /></div>
        </div>

        <details className="rounded-xl border bg-card p-4"><summary className="cursor-pointer text-sm font-medium">Downloads and storage{downloads.some((download) => download.active) ? ` · ${downloads.filter((download) => download.active).length} active` : ''}</summary><div className="mt-4"><DownloadPanel location={data?.location} /></div></details>

        {isLoading ? <div className="flex justify-center p-12"><LoadingSpinner size="lg" /></div> : error ? <div role="alert" className="rounded-xl border border-destructive/40 bg-destructive/5 p-6"><h2 className="font-semibold">Could not load the model library</h2><p className="mt-2 text-sm text-muted-foreground">{formatApiError(error, 'Check the connection to your analysis server.')}</p><button onClick={() => refetch()} className={`${ACTION} mt-4`}>Try again</button></div> : filtered.length > 0 ? <div className="grid items-stretch gap-4 lg:grid-cols-2">{filtered.map((model) => <ModelCard key={model.id} model={model} allModels={models} organization={organization} download={downloads.find((d) => d.repo_id === model.model_ref)} onViewHistory={viewHistory} />)}</div> : <div className="rounded-xl border border-dashed p-10 text-center"><h2 className="text-lg font-semibold">{search || experimentId ? 'No matching models' : kind === 'custom' ? 'No custom checkpoints on this server yet' : 'No models found'}</h2><p className="mx-auto mt-2 max-w-lg text-sm text-muted-foreground">{search || experimentId ? 'Try another name, clear the filters, or assign models to this experiment using Organize.' : kind === 'custom' ? 'Save an edited checkpoint from Neurosurgery to add it here. Existing checkpoints in the server model directory appear automatically.' : 'Refresh the library or enter a model ID in Interpretability to begin.'}</p>{search || experimentId ? <button className={`${ACTION} mt-4`} onClick={() => { setKind('all'); updateWorkspace({ experiment: '', clearSearch: true }) }}>Clear filters</button> : <Link className={`${ACTION} mt-4`} href={kind === 'custom' ? '/weights' : '/interp'}>{kind === 'custom' ? 'Open Neurosurgery' : 'Start an analysis'}<ArrowRight className="h-4 w-4" /></Link>}</div>}
        </div>}
      </div>
    </Layout>
  )
}
