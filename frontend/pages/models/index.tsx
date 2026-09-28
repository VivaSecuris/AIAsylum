import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { MessageSquare, GitCompare, RefreshCw, Search, Scissors, ArrowRight } from 'lucide-react'
import { Layout } from '@/components/layout/Layout'
import { apiClient } from '@/lib/api'
import { useModelCatalog } from '@/lib/model-catalog'
import { useTestRuns } from '@/lib/hooks'
import { getSettings, type AppSettings } from '@/lib/settings'
import { modelChatUrl, supportsWeightActions } from '@/lib/model-chat'
import { buildModelOverview, modelComparisonUrl, type OverviewModel } from '@/lib/model-overview'
import { formatApiError } from '@/lib/utils'

const BUTTON = 'inline-flex items-center justify-center gap-2 rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

export function OverviewModelCard({ model, selected, onSelect }: { model: OverviewModel; selected: boolean; onSelect: () => void }) {
  return <article className="flex flex-col rounded-xl border bg-card p-5 shadow-sm">
    <div className="mb-3 flex items-start justify-between gap-3">
      <span className="text-xs text-muted-foreground">{model.providerLabel} · {model.custom ? 'Modified checkpoint' : model.kind === 'checkpoint' ? 'Base checkpoint' : model.kind === 'api' ? 'Hosted model' : 'Model service'}</span>
      <label className="flex shrink-0 items-center gap-1.5 text-xs"><input type="checkbox" checked={selected} onChange={onSelect} aria-label={`Compare ${model.provider} / ${model.model}`} />Compare</label>
    </div>
    <h2 className="break-words text-lg font-semibold"><Link href={modelChatUrl(model.provider, model.model)} className="hover:text-primary hover:underline">{model.name}</Link></h2>
    <p className="mt-1 break-all text-xs text-muted-foreground">{model.provider} / {model.model}</p>
    <p className={`mt-3 text-sm ${model.ready ? 'text-emerald-700 dark:text-emerald-400' : 'text-muted-foreground'}`}>{model.status}</p>
    {model.reason && <p className="mt-1 text-xs text-muted-foreground">{model.reason}</p>}
    <p className="mt-2 text-xs text-muted-foreground">{model.runs ? `${model.runs} recorded test run${model.runs === 1 ? '' : 's'}` : 'No test runs recorded'}</p>
    <div className="mt-auto pt-4">
      <Link href={modelChatUrl(model.provider, model.model)} className={`${BUTTON} w-full border-primary bg-primary text-primary-foreground hover:bg-primary/90`}><MessageSquare className="h-4 w-4" />Chat</Link>
      <div className="mt-2 flex flex-wrap gap-2">
        <Link href={{ pathname: '/create-test', query: { provider: model.provider, model: model.model } }} className={`${BUTTON} flex-1`}>Run tests</Link>
        {supportsWeightActions(model.provider) && <Link href={{ pathname: '/weights', query: { source_model: model.model } }} className={`${BUTTON} flex-1`}><Scissors className="h-4 w-4" />Modify</Link>}
        <Link href={modelComparisonUrl([model])} className={BUTTON}>Compare</Link>
      </div>
      {model.catalog && <Link href={{ pathname: '/compare', query: { view: 'history', model: model.model } }} className="mt-3 inline-block text-xs text-primary hover:underline">Checkpoint history &amp; details →</Link>}
    </div>
  </article>
}

export default function ModelsOverviewPage() {
  const catalog = useModelCatalog()
  const providers = useQuery({ queryKey: ['providers'], queryFn: () => apiClient.listProviders(), staleTime: 15000 })
  const ollama = useQuery({ queryKey: ['ollama-models'], queryFn: () => apiClient.listOllamaModels(), staleTime: 15000 })
  const runs = useTestRuns({ limit: 500 })
  const [defaults, setDefaults] = useState<AppSettings>()
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [selected, setSelected] = useState<string[]>([])
  useEffect(() => { setDefaults(getSettings()) }, [])
  const models = useMemo(() => buildModelOverview(catalog.data?.models || [], providers.data || [], ollama.data, runs.data || [], defaults), [catalog.data, providers.data, ollama.data, runs.data, defaults])
  const visible = models.filter(model => (filter === 'all' || filter === 'ready' && model.ready || filter === 'custom' && model.custom || model.kind === filter) && [model.name, model.model, model.providerLabel, model.provider].some(text => text.toLowerCase().includes(search.trim().toLowerCase())))
  const comparison = models.filter(model => selected.includes(model.key))
  const loading = catalog.isLoading || providers.isLoading || ollama.isLoading
  const errors = [catalog.error, providers.error, ollama.error, runs.error].filter(Boolean)

  function toggle(key: string) {
    setSelected(current => current.includes(key) ? current.filter(value => value !== key) : current.length < 20 ? [...current, key] : current)
  }
  return <Layout><div className="mx-auto max-w-7xl space-y-6">
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div><h1 className="text-3xl font-bold">Models</h1><p className="mt-2 text-sm text-muted-foreground">Choose a model and start chatting. Use prompts, tests, and comparisons to explore its behavior.</p></div>
      <div className="flex flex-wrap gap-2"><Link href="/models/chat" className={BUTTON}>Open another model</Link><Link href="/compare?view=find" className={BUTTON}>Find or download models<ArrowRight className="h-4 w-4" /></Link></div>
    </div>
    <div className="grid grid-cols-3 gap-3">
      {[[models.length, 'Known models'], [models.filter(model => model.ready).length, 'Installed or configured'], [models.filter(model => model.custom).length, 'Modified checkpoints']].map(([count, label]) => <div key={label} className="rounded-lg border bg-card px-4 py-3"><div className="text-2xl font-semibold">{count}</div><p className="text-xs text-muted-foreground">{label}</p></div>)}
    </div>
    {errors.length > 0 && <div role="alert" className="space-y-1 rounded-lg border border-amber-300 p-3 text-sm"><p>Some model sources could not be loaded. Available results are shown below.</p>{errors.map((error, index) => <p key={index} className="text-xs">{formatApiError(error, 'Model source unavailable.')}</p>)}</div>}
    <div className="flex flex-wrap items-center gap-3">
      <div className="relative min-w-56 flex-1"><Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" /><input aria-label="Search models" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search models or providers…" className="w-full rounded-md border bg-background py-2 pl-9 pr-3 text-sm" /></div>
      <select aria-label="Filter model inventory" value={filter} onChange={event => setFilter(event.target.value)} className="rounded-md border bg-background px-3 py-2 text-sm"><option value="all">All models</option><option value="ready">Installed or configured</option><option value="custom">Modified checkpoints</option><option value="checkpoint">Local checkpoints</option><option value="service">Model services</option><option value="api">Hosted models</option></select>
      <button className={BUTTON} aria-label="Refresh models" onClick={() => { void catalog.refetch(); void providers.refetch(); void ollama.refetch(); void runs.refetch() }}><RefreshCw className="h-4 w-4" /></button>
    </div>
    {comparison.length > 0 && <div className="flex flex-wrap items-center gap-3 rounded-lg border border-primary/30 bg-primary/5 p-4"><p className="flex-1 text-sm">{comparison.length} selected · Use the same prompts and settings for each model.</p>{comparison.length >= 2 ? <Link href={modelComparisonUrl(comparison)} className={BUTTON}><GitCompare className="h-4 w-4" />Compare selected models</Link> : <span className="text-xs text-muted-foreground">Select another model to compare.</span>}<button className="text-xs underline" onClick={() => setSelected([])}>Clear selection</button>{comparison.length === 20 && <p className="w-full text-xs">Up to 20 models per selection.</p>}</div>}
    {loading && <p role="status" className="text-sm text-muted-foreground">Loading model inventory…</p>}
    {visible.length ? <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{visible.map(model => <OverviewModelCard key={model.key} model={model} selected={selected.includes(model.key)} onSelect={() => toggle(model.key)} />)}</div> : !loading && <p className="rounded-lg border border-dashed p-10 text-center text-muted-foreground">No matching models. Clear the filter, find a model, or open one by its provider and name.</p>}
    <section className="rounded-xl border bg-card p-5"><h2 className="font-semibold">Prompts and evaluation</h2><p className="mt-1 text-sm text-muted-foreground">Choose instructions and measurements for the model you want to explore.</p><div className="mt-3 flex flex-wrap gap-2"><Link href="/prompts" className={BUTTON}>Prompt library</Link><Link href="/benchmarks" className={BUTTON}>Benchmarks</Link><Link href="/compare?view=charts" className={BUTTON}>Comparison charts</Link><Link href="/test-runs" className={BUTTON}>Test history</Link></div></section>
  </div></Layout>
}
