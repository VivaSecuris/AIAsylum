import { useMemo, useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, ArrowRight, Check, Search } from 'lucide-react'
import { useModelCatalog, editDescription, type ModelCatalogEntry } from '@/lib/model-catalog'
import { modelOrganizationKey, useModelOrganization } from '@/lib/model-organization'
import { BENCHMARK_CHECKS, benchmarkName, shortModelName, type BenchmarkCampaignRequest } from '@/lib/benchmark-campaigns'

const field = 'mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm'
const primary = 'inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50'
const secondary = 'inline-flex items-center justify-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50'

export function CampaignBuilder({ onStart, pending, error, initial }: {
  onStart: (request: BenchmarkCampaignRequest) => void
  pending: boolean
  error: string | null
  initial?: BenchmarkCampaignRequest
}) {
  const { data: catalog, isLoading, error: catalogError } = useModelCatalog()
  const { data: organization } = useModelOrganization()
  const [stage, setStage] = useState(0)
  const [models, setModels] = useState<string[]>(initial?.models ?? [])
  const [checks, setChecks] = useState<string[]>(initial?.benchmarks ?? ['mmlu', 'gsm8k', 'hellaswag', 'arc'])
  const [samples, setSamples] = useState(initial?.num_samples ?? 25)
  const [seed, setSeed] = useState(initial?.seed ?? 0)
  const [tokens, setTokens] = useState(initial?.max_new_tokens ?? 256)
  const [name, setName] = useState(initial ? `${initial.name.slice(0, 112)} · retry` : '')
  const [search, setSearch] = useState('')
  const ready = useMemo(() => (catalog?.models ?? []).filter((model) => model.availability === 'ready'), [catalog])
  const readyRefs = new Set(ready.map((model) => model.model_ref))
  const unavailable = models.filter((ref) => !readyRefs.has(ref))
  const label = (model: ModelCatalogEntry) => organization?.items[modelOrganizationKey(model.model_ref)]?.label || model.name
  const visible = ready.filter((model) => `${label(model)} ${model.model_ref} ${model.source_model ?? ''}`.toLowerCase().includes(search.toLowerCase()))
  const validModels = models.length > 0 && models.length <= 30 && unavailable.length === 0
  const validChecks = checks.length > 0 && Number.isInteger(samples) && samples >= 1 && samples <= 200 && Number.isInteger(seed) && seed >= 0 && seed <= 2147483647 && Number.isInteger(tokens) && tokens >= 16 && tokens <= 2048
  const toggle = (ref: string) => setModels((current) => current.includes(ref) ? current.filter((item) => item !== ref) : [...current, ref])
  const totalRuns = models.length * checks.length
  const request: BenchmarkCampaignRequest = { name: name.trim() || `${models.length} models · ${checks.map(benchmarkName).join(', ')}`, models, benchmarks: checks, num_samples: samples, seed, max_new_tokens: tokens }

  return <section aria-label="New benchmark comparison" className="overflow-hidden rounded-xl border bg-card">
    <ol className="grid grid-cols-3 border-b bg-muted/30">
      {['Choose models', 'Choose checks', 'Review & run'].map((title, index) => <li key={title}>
        <button type="button" aria-current={stage === index ? 'step' : undefined} disabled={pending || (index > stage && (!validModels || (index === 2 && !validChecks)))} onClick={() => setStage(index)} className={`flex w-full items-center justify-center gap-2 px-3 py-4 text-sm ${stage === index ? 'bg-primary/10 font-semibold text-primary' : 'text-muted-foreground'} disabled:opacity-50`}>
          <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs ${stage === index ? 'bg-primary text-primary-foreground' : 'border'}`}>{stage > index ? <Check className="h-3 w-3" /> : index + 1}</span>
          <span>{title}</span>
        </button>
      </li>)}
    </ol>

    <div className="space-y-5 p-5 sm:p-6">
      {stage === 0 && <>
        <div><h2 className="text-xl font-semibold">Which models do you want to compare?</h2><p className="mt-1 text-sm text-muted-foreground">Choose an original model and its custom versions, or compare different model sizes. Models below are ready on this server.</p></div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative min-w-64 flex-1"><Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" /><input aria-label="Search models to benchmark" placeholder="Search models, families, or custom checkpoints" value={search} onChange={(event) => setSearch(event.target.value)} className="w-full rounded-lg border bg-background py-2 pl-9 pr-3 text-sm" /></div>
          <Link href="/compare?view=find" className={secondary}>Find & download models <ArrowRight className="h-4 w-4" /></Link>
        </div>
        {isLoading && <p role="status" className="text-sm text-muted-foreground">Loading models on this server…</p>}
        {catalogError && <p role="alert" className="rounded-lg border border-destructive/30 p-3 text-sm text-destructive">Could not load the model library. Check the server connection and try again.</p>}
        {!isLoading && !catalogError && !ready.length && <p className="rounded-lg border p-4 text-sm">No complete checkpoints are available yet. Find a model above and finish its download, then return here.</p>}
        <div className="grid gap-4 lg:grid-cols-2">
          {(['base', 'custom'] as const).map((kind) => {
            const group = visible.filter((model) => model.kind === kind)
            return <div key={kind} className="rounded-lg border">
              <div className="flex items-center justify-between border-b px-4 py-3"><h3 className="font-medium">{kind === 'base' ? 'Original models' : 'Our custom models'} <span className="text-sm text-muted-foreground">({group.length})</span></h3><button type="button" disabled={!group.length || pending} onClick={() => setModels((current) => Array.from(new Set([...current, ...group.map((model) => model.model_ref)])).slice(0, 30))} className="text-xs text-primary hover:underline disabled:opacity-50">Select visible</button></div>
              <div className="max-h-80 overflow-y-auto divide-y">
                {group.map((model) => <div key={model.model_ref} className={models.includes(model.model_ref) ? 'bg-primary/5' : ''}>
                  <label className="flex cursor-pointer items-start gap-3 px-4 py-3">
                    <input type="checkbox" checked={models.includes(model.model_ref)} disabled={pending || (models.length >= 30 && !models.includes(model.model_ref))} onChange={() => toggle(model.model_ref)} className="mt-1 h-4 w-4 shrink-0 accent-primary" />
                    <span className="min-w-0"><span className="block break-words text-sm font-medium">{label(model)}</span><span className="mt-0.5 block break-all text-xs text-muted-foreground">{model.model_ref}</span>{kind === 'custom' && <span className="mt-1 block text-xs text-muted-foreground">{editDescription(model)}</span>}</span>
                  </label>
                  {kind === 'custom' && models.includes(model.model_ref) && model.source_model && readyRefs.has(model.source_model) && !models.includes(model.source_model) && <button type="button" disabled={models.length >= 30} className="mb-3 ml-11 text-xs text-primary hover:underline disabled:opacity-50" onClick={() => setModels((current) => [...current, model.source_model!])}>Include original: {shortModelName(model.source_model)}</button>}
                </div>)}
                {!group.length && <p className="p-4 text-sm text-muted-foreground">{kind === 'custom' ? 'No matching custom checkpoints on this server.' : 'No matching original models.'}</p>}
              </div>
            </div>
          })}
        </div>
        {!!models.length && <div className="rounded-lg bg-muted/40 p-3"><div className="flex items-center justify-between"><p className="text-sm font-medium">{models.length} model{models.length !== 1 ? 's' : ''} selected</p><button type="button" onClick={() => setModels([])} className="text-xs text-primary hover:underline">Clear selection</button></div><div className="mt-2 flex flex-wrap gap-2">{models.map((ref) => <button key={ref} type="button" onClick={() => toggle(ref)} title={`Remove ${ref}`} className="max-w-full break-all rounded border bg-background px-2 py-1 text-xs">{shortModelName(ref)} ×</button>)}</div></div>}
        {!!unavailable.length && !isLoading && <p role="alert" className="text-sm text-destructive">{unavailable.length} selected checkpoint(s) are no longer ready on this server. Remove them or finish their download.</p>}
        {models.length === 1 && <p className="text-xs text-muted-foreground">One model gives you a baseline. Add a second model to compare results.</p>}
      </>}

      {stage === 1 && <>
        <div><h2 className="text-xl font-semibold">What should every model be tested on?</h2><p className="mt-1 text-sm text-muted-foreground">Each selected model receives the same sampled questions and generation settings for each check.</p></div>
        <div className="grid gap-3 sm:grid-cols-2">{BENCHMARK_CHECKS.map((check) => <label key={check.id} className={`flex cursor-pointer gap-3 rounded-lg border p-4 ${checks.includes(check.id) ? 'border-primary bg-primary/5' : ''}`}><input type="checkbox" checked={checks.includes(check.id)} onChange={() => setChecks((current) => current.includes(check.id) ? current.filter((item) => item !== check.id) : [...current, check.id])} className="mt-1 h-4 w-4 accent-primary" /><span><span className="block font-medium">{check.name}</span><span className="mt-1 block text-sm text-muted-foreground">{check.description}</span><span className="mt-2 block text-xs text-muted-foreground">{check.format}</span></span></label>)}</div>
        <div className="grid gap-4 sm:grid-cols-3">
          <label className="text-sm font-medium">Questions per check<input type="number" min={1} max={200} step={1} value={Number.isNaN(samples) ? '' : samples} onChange={(event) => setSamples(event.target.valueAsNumber)} className={field} /><span className="mt-1 block text-xs font-normal text-muted-foreground">1–200 for each model</span></label>
          <label className="text-sm font-medium">Sample seed<input type="number" min={0} max={2147483647} step={1} value={Number.isNaN(seed) ? '' : seed} onChange={(event) => setSeed(event.target.valueAsNumber)} className={field} /><span className="mt-1 block text-xs font-normal text-muted-foreground">Use the same seed when retrying</span></label>
          <label className="text-sm font-medium">Maximum answer tokens<input type="number" min={16} max={2048} step={1} value={Number.isNaN(tokens) ? '' : tokens} onChange={(event) => setTokens(event.target.valueAsNumber)} className={field} /><span className="mt-1 block text-xs font-normal text-muted-foreground">16–2,048; longer answers take more time</span></label>
        </div>
        <p className="rounded-lg border bg-muted/30 p-3 text-sm text-muted-foreground">This is small sample screening, not a published leaderboard. Compare results within a check; different checks measure different skills.</p>
      </>}

      {stage === 2 && <>
        <div><h2 className="text-xl font-semibold">Review your comparison</h2><p className="mt-1 text-sm text-muted-foreground">Runs are saved together and processed one at a time on the server GPU. You can leave this page and return to the results.</p></div>
        <label className="block text-sm font-medium">Comparison name<input maxLength={120} value={name} onChange={(event) => setName(event.target.value)} placeholder={`${models.length} models · baseline comparison`} className={field} /></label>
        <dl className="grid gap-4 rounded-lg border p-4 sm:grid-cols-3"><div><dt className="text-xs text-muted-foreground">Saved runs</dt><dd className="mt-1 text-xl font-semibold">{totalRuns}</dd><dd className="text-xs text-muted-foreground">{models.length} models × {checks.length} checks</dd></div><div><dt className="text-xs text-muted-foreground">Requested answers</dt><dd className="mt-1 text-xl font-semibold">{(totalRuns * samples).toLocaleString()}</dd><dd className="text-xs text-muted-foreground">{samples} questions per check</dd></div><div><dt className="text-xs text-muted-foreground">Shared settings</dt><dd className="mt-1 text-sm">Seed {seed} · {tokens} answer tokens</dd><dd className="text-xs text-muted-foreground">Deterministic generation</dd></div></dl>
        <div className="grid gap-4 sm:grid-cols-2"><div><h3 className="text-sm font-medium">Models</h3><ul className="mt-2 max-h-48 space-y-1 overflow-y-auto text-sm text-muted-foreground">{models.map((ref) => <li key={ref} className="break-all">{ref}</li>)}</ul></div><div><h3 className="text-sm font-medium">Checks</h3><ul className="mt-2 space-y-1 text-sm text-muted-foreground">{checks.map((check) => <li key={check}>{benchmarkName(check)}</li>)}</ul></div></div>
      </>}

      {error && <p role="alert" className="rounded-lg border border-destructive/30 p-3 text-sm text-destructive">{error}</p>}
      <div className="flex items-center justify-between gap-3 border-t pt-4">
        {stage > 0 ? <button type="button" disabled={pending} onClick={() => setStage(stage - 1)} className={secondary}><ArrowLeft className="h-4 w-4" /> Back</button> : <span className="text-xs text-muted-foreground">{catalog?.location ? `Connected server: ${catalog.location}` : 'Server model library'}</span>}
        {stage < 2 ? <button type="button" disabled={!validModels || (stage === 1 && !validChecks)} onClick={() => setStage(stage + 1)} className={primary}>{stage === 0 ? 'Choose checks' : 'Review comparison'} <ArrowRight className="h-4 w-4" /></button> : <button type="button" disabled={pending || !validModels || !validChecks} onClick={() => onStart(request)} className={primary}>{pending ? 'Saving comparison…' : `Run ${totalRuns} benchmarks`} <ArrowRight className="h-4 w-4" /></button>}
      </div>
    </div>
  </section>
}
