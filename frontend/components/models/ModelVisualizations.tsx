import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { ArrowUpRight, BarChart3, Brain, Scissors } from 'lucide-react'
import { apiClient } from '@/lib/api'
import { useInterpRuns } from '@/lib/hooks'
import { formatApiError, parseApiDate } from '@/lib/utils'
import { useModelOrganization, modelOrganizationKey } from '@/lib/model-organization'
import { EvaluationCharts } from './EvaluationCharts'
import { WeightVisualizations } from './WeightVisualizations'

const ACTION = 'inline-flex items-center justify-center gap-2 rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
const MODES: Record<string, string> = { single: 'Single prompt', comparison: 'Prompt comparison', model_diff: 'Model comparison', multi_prompt: 'Few-shot progression' }

export function ModelVisualizations() {
  const router = useRouter()
  const { data: runs = [], isLoading, error, refetch } = useInterpRuns({ limit: 1000 })
  const { data: organization } = useModelOrganization()
  const completed = runs.filter((run) => run.status === 'completed').sort((a, b) => b.id - a.id)
  const visual = router.query.visual === 'interpretability' || router.query.visual === 'weights' ? router.query.visual : 'evaluation'
  const selectedId = typeof router.query.analysis === 'string' ? Number(router.query.analysis) : completed[0]?.id
  const selected = completed.find((run) => run.id === selectedId)
  const [search, setSearch] = useState('')
  const [frameLoading, setFrameLoading] = useState(true)
  useEffect(() => { setFrameLoading(true) }, [selected?.id])
  const modelName = (ref: string) => organization?.items[modelOrganizationKey(ref)]?.label || ref.split('/').filter(Boolean).pop() || ref
  const matching = completed.filter((run) => [String(run.id), run.model_a, run.model_b, modelName(run.model_a), run.model_b ? modelName(run.model_b) : '', MODES[run.mode]].some((part) => part?.toLowerCase().includes(search.trim().toLowerCase())))
  function navigate(values: Record<string, string>) {
    void router.replace({ pathname: '/compare', query: { ...router.query, view: 'charts', ...values } }, undefined, { shallow: true, scroll: false })
  }
  return <div className="space-y-5">
    <div>
      <h2 className="text-xl font-semibold">Charts &amp; visualizations</h2>
      <p className="mt-1 text-sm text-muted-foreground">Explore evaluation comparisons, saved layer dashboards, and weight experiment plots. These views use results recorded on the connected server.</p>
      <Link href="/benchmarks" className={`${ACTION} mt-3`}><BarChart3 className="h-4 w-4" />Benchmark scores &amp; comparisons<ArrowUpRight className="h-4 w-4" /></Link>
    </div>
    <div className="grid gap-3 md:grid-cols-3" role="tablist" aria-label="Visualization type">
      {([
        ['evaluation', 'Evaluation comparisons', 'Scores, radar charts, test counts and model statistics', BarChart3],
        ['interpretability', 'Interpretability dashboards', `${isLoading ? 'Loading' : completed.length} saved dashboards · activations, predictions and comparisons`, Brain],
        ['weights', 'Weight experiment plots', 'Layer separation, steering curves and capability tradeoffs', Scissors],
      ] as const).map(([id, name, description, Icon]) => <button key={id} id={`visual-${id}-tab`} type="button" role="tab" aria-selected={visual === id} aria-controls={`visual-${id}-panel`} onClick={() => navigate({ visual: id })} className={`rounded-xl border p-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${visual === id ? 'border-primary bg-primary/5 ring-1 ring-primary' : 'bg-card hover:bg-muted/30'}`}><span className="flex items-center gap-2 text-sm font-semibold"><Icon className="h-4 w-4" />{name}</span><span className="mt-2 block text-xs text-muted-foreground">{description}</span></button>)}
    </div>
    <div id={`visual-${visual}-panel`} role="tabpanel" aria-labelledby={`visual-${visual}-tab`}>
      {visual === 'evaluation' ? <EvaluationCharts /> : visual === 'weights' ? <WeightVisualizations /> : <section aria-label="Saved interpretability dashboards" className="space-y-4">
        {error ? <div role="alert" className="rounded-lg border p-4 text-sm text-destructive">{formatApiError(error, 'Could not load saved dashboards.')} <button type="button" onClick={() => refetch()} className="underline">Retry</button></div> : isLoading ? <p role="status" className="p-6 text-sm">Loading saved dashboards…</p> : completed.length === 0 ? <div className="rounded-xl border border-dashed p-6"><h3 className="font-semibold">No completed interpretability dashboards on this server yet</h3><p className="mt-2 text-sm text-muted-foreground">A completed analysis saves its interactive plots here. Imported history records do not include dashboard files.</p><Link href="/interp" className={`${ACTION} mt-4`}>Prepare an analysis</Link></div> : <>
          <div className="grid items-end gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
            <div><label htmlFor="dashboard-search" className="text-xs font-medium">Find a dashboard</label><input id="dashboard-search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Model, comparison or run number…" className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm" /></div>
            <div><label htmlFor="saved-dashboard" className="text-xs font-medium">Saved analysis</label><select id="saved-dashboard" value={selected?.id ?? ''} onChange={(event) => navigate({ analysis: event.target.value })} className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"><option value="" disabled>Choose a saved analysis</option>{selected && !matching.some((run) => run.id === selected.id) && <option value={selected.id}>#{selected.id} · {modelName(selected.model_a)} · Current selection</option>}{matching.map((run) => <option key={run.id} value={run.id}>#{run.id} · {MODES[run.mode] || run.mode} · {modelName(run.model_a)}{run.model_b ? ` → ${modelName(run.model_b)}` : ''}</option>)}</select></div>
          </div>
          {search && matching.length === 0 && <p className="text-sm text-muted-foreground">No dashboards match that search. Clear the search to see all saved analyses.</p>}
          {selected ? <>
            <div className="flex flex-wrap items-start justify-between gap-3 rounded-lg border bg-card p-4">
              <div><h3 className="font-semibold">Analysis #{selected.id} · {MODES[selected.mode] || selected.mode}</h3><p className="mt-1 break-all text-xs text-muted-foreground">{selected.model_a}{selected.model_b ? ` → ${selected.model_b}` : ''}</p><p className="mt-1 text-xs text-muted-foreground">{selected.created_at ? parseApiDate(selected.created_at).toLocaleString() : 'Time not recorded'} · {selected.metadata?.summary?.preflight?.device || selected.metadata?.options?.device || 'Device not recorded'}</p></div>
              <div className="flex flex-wrap gap-2"><Link href={`/interp/${selected.id}`} className={ACTION}>Full analysis &amp; causal tests <ArrowUpRight className="h-4 w-4" /></Link><a href={apiClient.interpDashboardUrl(selected.id)} target="_blank" rel="noreferrer" className={ACTION}>Open full dashboard <ArrowUpRight className="h-4 w-4" /></a></div>
            </div>
            {frameLoading && <p role="status" className="text-xs text-muted-foreground">Loading interactive charts…</p>}
            <iframe key={selected.id} src={apiClient.interpDashboardUrl(selected.id)} onLoad={() => setFrameLoading(false)} title={`Saved interpretability dashboard ${selected.id}`} className="h-[850px] w-full rounded-xl border bg-white" />
          </> : <p role="alert" className="rounded-lg border p-4 text-sm">That analysis is unavailable. Choose a completed analysis from the list.</p>}
        </>}
      </section>}
    </div>
  </div>
}
