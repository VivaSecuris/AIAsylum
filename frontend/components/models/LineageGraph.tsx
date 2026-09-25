import { useEffect, useId, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import { ArrowUpRight, GitBranch, Maximize2, Minus, Plus, RefreshCw } from 'lucide-react'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useModelLineage, type ModelLineageNode } from '@/lib/model-catalog'
import { formatApiError, parseApiDate } from '@/lib/utils'
import { OrganizationEditor } from './OrganizationEditor'
import { filterExperimentLineage, stepOrganizationKey, useModelOrganization } from '@/lib/model-organization'
import { COLUMN_GAP, layoutLineage, NODE_HEIGHT, NODE_WIDTH, relatedLineage } from './lineage-layout'

const BUTTON = 'inline-flex items-center justify-center gap-2 rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
const KIND_LABELS: Record<ModelLineageNode['kind'], string> = { model: 'Checkpoint', direction: 'Direction', sweep: 'Causal check', select: 'Capability search', surgery: 'Weight edit', expert_surgery: 'Expert edit', routing: 'Expert routing', lora: 'LoRA fine-tune', distill: 'Distillation', compare: 'Behavior comparison', probe: 'Intent probe', interp: 'Analysis', test: 'Evaluation', missing: 'Missing step' }

function dateLabel(value: string | null, full = false) {
  if (!value) return 'Time not recorded'
  const date = parseApiDate(value)
  if (Number.isNaN(date.getTime())) return value
  return full ? date.toLocaleString() : date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

function statusLabel(status: string) { return status.replace(/_/g, ' ') }
function statusClass(node: ModelLineageNode) {
  if (node.kind === 'missing' || ['missing', 'deleted', 'incomplete', 'failed', 'cancelled'].includes(node.status)) return 'border-amber-400 bg-amber-50 dark:bg-amber-950/30'
  if (['running', 'pending'].includes(node.status)) return 'border-blue-400 bg-blue-50 dark:bg-blue-950/30'
  return node.kind === 'model' ? 'border-primary/50 bg-background' : 'border-border bg-card'
}

function JsonDetails({ title, value, empty }: { title: string; value: Record<string, unknown>; empty: string }) {
  const present = value && Object.keys(value).length > 0
  return <details className="rounded-lg border p-3"><summary className="cursor-pointer text-sm font-medium">{title}</summary>{present ? <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/40 p-3 text-xs">{JSON.stringify(value, null, 2)}</pre> : <p className="mt-2 text-xs text-muted-foreground">{empty}</p>}</details>
}

function OutcomeHighlights({ node }: { node: ModelLineageNode }) {
  const result = node.summary as Record<string, any>
  const values: Array<[string, unknown]> = [
    ['Direction AUC', result.auc], ['Layer', result.layer], ['Rank', result.rank],
    ['Edit strength (β)', result.manifest?.beta],
    ['Hidden-state snapshots', node.kind === 'interp' ? result.num_layers : undefined],
    ['Tokens compared', node.kind === 'interp' ? result.window_len : undefined],
    ['Factual accuracy change', result.deltas?.factual_acc],
    ['Refusal rate change', result.deltas?.refuse_harmful],
    ['Weight experiments', result.history?.weight_runs], ['Analysis runs', result.history?.interp_runs],
    ['Test runs', result.history?.test_runs],
  ]
  const visible = values.filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
  return visible.length > 0 ? <dl className="grid grid-cols-2 gap-2">{visible.map(([label, value]) => <div key={label} className="rounded-lg bg-muted/50 p-3"><dt className="text-xs text-muted-foreground">{label}</dt><dd className="mt-1 text-lg font-semibold">{Number.isInteger(value) ? String(value) : Number(value).toFixed(3)}</dd></div>)}</dl> : null
}

function SettingsHighlights({ node }: { node: ModelLineageNode }) {
  const settings = node.settings as Record<string, any>
  const options = settings.options ?? {}
  const manifest = settings.manifest ?? {}
  const editsWeights = ['surgery', 'expert_surgery', 'lora', 'distill', 'model'].includes(node.kind)
  const values: Array<[string, unknown]> = [
    ['Method', settings.method ?? manifest.method], ['Objective', settings.objective ?? manifest.objective],
    ['Rank', options.subspace_rank ?? options.rfm_rank ?? options.rank ?? manifest.rank],
    ['Edit strength (β)', editsWeights ? options.beta ?? manifest.beta : undefined],
    ['Seed', options.seed], ['Examples per class', node.kind === 'direction' ? options.n_per_class : undefined],
    ['Evaluation prompts', ['compare', 'sweep', 'select'].includes(node.kind) ? options.n_prompts : undefined],
    ['Sequence limit', options.max_len], ['Analysis window', options.window],
    ['Device', options.device], ['Precision', options.dtype],
  ]
  const visible = values.filter(([, value]) => ['string', 'number'].includes(typeof value))
  return visible.length > 0 ? <dl className="grid grid-cols-2 gap-x-3 gap-y-2 rounded-lg border p-3 text-xs">{visible.map(([label, value]) => <div key={label}><dt className="text-muted-foreground">{label}</dt><dd className="mt-1 break-words font-medium">{String(value).replace(/_/g, ' ')}</dd></div>)}</dl> : null
}

export function LineageGraph({ focusModel, onFocusModelChange, experimentId = '' }: { focusModel?: string; onFocusModelChange?: (modelRef: string | undefined) => void; experimentId?: string }) {
  const { data, isLoading, error, refetch, isFetching } = useModelLineage()
  const { data: organization } = useModelOrganization()
  const [presentation, setPresentation] = useState<'graph' | 'steps'>('graph')
  const [stepSearch, setStepSearch] = useState('')
  const [stepKind, setStepKind] = useState('')
  const [scope, setScope] = useState(focusModel ?? '')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [zoom, setZoom] = useState(0.85)
  const viewport = useRef<HTMLDivElement>(null)
  const markerId = `lineage-arrow-${useId().replace(/:/g, '')}`
  const namedNodes = useMemo(() => (data?.nodes ?? []).map((node) => ({ ...node, label: organization?.items[stepOrganizationKey(node)]?.label || node.label })), [data, organization])
  const graph = useMemo(() => filterExperimentLineage(relatedLineage(namedNodes, data?.edges ?? [], scope), organization, experimentId), [namedNodes, data, scope, organization, experimentId])
  const steps = useMemo(() => {
    const needle = stepSearch.trim().toLowerCase()
    const timestamp = (node: ModelLineageNode) => {
      const value = node.created_at ? parseApiDate(node.created_at).getTime() : NaN
      return Number.isFinite(value) ? value : Number.MAX_SAFE_INTEGER
    }
    return graph.nodes.filter((node) => (!stepKind || node.kind === stepKind) && (!needle || [node.label, node.model_ref, node.status, KIND_LABELS[node.kind], organization?.items[stepOrganizationKey(node)]?.notes].some((value) => value?.toLowerCase().includes(needle)))).sort((a, b) => timestamp(a) - timestamp(b) || a.id.localeCompare(b.id))
  }, [graph, stepSearch, stepKind, organization])
  const layout = useMemo(() => layoutLineage(graph.nodes, graph.edges), [graph])
  const selected = namedNodes.find((node) => node.id === selectedId && graph.nodes.some((visible) => visible.id === node.id))
  const models = useMemo(() => Array.from(new Map(namedNodes.filter((node) => node.kind === 'model' && node.model_ref).map((node) => [node.model_ref!, node])).values()).sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true })), [namedNodes])
  const selectedItem = selected ? organization?.items[stepOrganizationKey(selected)] : undefined
  const recordedLabel = selected ? data?.nodes.find((node) => node.id === selected.id)?.label : undefined
  const incoming = selected ? layout.edges.filter((edge) => edge.target === selected.id) : []
  const outgoing = selected ? layout.edges.filter((edge) => edge.source === selected.id) : []

  useEffect(() => { setScope(focusModel ?? '') }, [focusModel])
  useEffect(() => {
    const focus = graph.nodes.find((node) => node.kind === 'model' && node.model_ref === scope)
    setSelectedId(focus?.id ?? null)
    if (viewport.current) { viewport.current.scrollLeft = 0; viewport.current.scrollTop = 0 }
  }, [scope])
  useEffect(() => {
    if (!selectedId && scope && data) setSelectedId(data.nodes.find((node) => node.kind === 'model' && node.model_ref === scope)?.id ?? null)
  }, [data, scope, selectedId])
  useEffect(() => {
    const position = selectedId ? layout.positions.get(selectedId) : undefined
    const canvas = viewport.current
    if (!position || !canvas || presentation !== 'graph') return
    const left = position.x * zoom, top = position.y * zoom
    if (left < canvas.scrollLeft || left + NODE_WIDTH * zoom > canvas.scrollLeft + canvas.clientWidth || top < canvas.scrollTop || top + NODE_HEIGHT * zoom > canvas.scrollTop + canvas.clientHeight) {
      canvas.scrollTo({ left: Math.max(0, left - (canvas.clientWidth - NODE_WIDTH * zoom) / 2), top: Math.max(0, top - 48) })
    }
  }, [selectedId, layout, zoom, presentation])

  function changeScope(modelRef: string) {
    setScope(modelRef)
    onFocusModelChange?.(modelRef || undefined)
  }

  function selectNode(id: string, scroll = false) {
    setSelectedId(id)
    if (scroll) {
      const position = layout.positions.get(id)
      if (position && viewport.current) viewport.current.scrollTo({ left: Math.max(0, position.x * zoom - 40), top: Math.max(0, position.y * zoom - 80), behavior: 'smooth' })
    }
  }
  function fit() {
    if (!viewport.current) return
    setZoom(Math.max(0.25, Math.min(1, (viewport.current.clientWidth - 24) / layout.width, (viewport.current.clientHeight - 24) / layout.height)))
    viewport.current.scrollTo({ left: 0, top: 0 })
  }

  if (isLoading) return <div className="flex justify-center p-12"><LoadingSpinner size="lg" /></div>
  if (error) return <div role="alert" className="rounded-xl border border-destructive/40 p-6"><h2 className="font-semibold">Could not load model history</h2><p className="mt-2 text-sm text-muted-foreground">{formatApiError(error, 'Check the connection to the analysis server.')}</p><button onClick={() => refetch()} className={`${BUTTON} mt-4`}>Try again</button></div>

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="max-w-lg flex-1"><label htmlFor="lineage-scope" className="text-sm font-medium">History to show</label><select id="lineage-scope" value={scope} onChange={(event) => changeScope(event.target.value)} className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"><option value="">All model history</option>{scope && !models.some((node) => node.model_ref === scope) && <option value={scope}>{scope}</option>}{models.map((node) => <option key={node.model_ref} value={node.model_ref!}>{node.label} · ancestors and descendants</option>)}</select></div>
        <button type="button" onClick={() => refetch()} disabled={isFetching} className={BUTTON}><RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} /> Refresh history</button>
      </div>
      {experimentId && <p className="rounded-lg bg-muted/40 p-3 text-xs text-muted-foreground">Showing assigned steps with their recorded ancestors. Cards marked Context explain where an assigned step came from.</p>}
      <p className="text-sm text-muted-foreground">Follow recorded dependencies from left to right. Each checkpoint and experiment is a separate step. Select a step to inspect its settings and outcome, or branch into a new run.</p>
      {(data?.warnings ?? []).length > 0 && <details className="rounded-lg border border-amber-300 bg-amber-50/40 p-3 dark:bg-amber-950/20"><summary className="cursor-pointer text-sm font-medium">History notes ({data!.warnings.length})</summary><ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-muted-foreground">{data!.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details>}
      {layout.cyclic.length > 0 && <p role="alert" className="rounded-lg border border-amber-400 p-3 text-sm">Some recorded relationships form a cycle. Those steps are grouped at the right; their order cannot be determined from the saved history.</p>}
      {graph.nodes.length === 0 ? <div className="rounded-xl border border-dashed p-10 text-center"><h2 className="font-semibold">{experimentId ? 'No steps assigned in this history' : scope ? 'No history found for this checkpoint' : 'No model history recorded yet'}</h2><p className="mt-2 text-sm text-muted-foreground">{experimentId ? 'Choose All experiments above, then use Organize on a model or history step to add it to this experiment.' : scope ? 'Show all history to explore the available checkpoints and experiments.' : 'Saved checkpoints and recorded experiments will appear here as you work.'}</p>{scope && <button className={`${BUTTON} mt-4`} onClick={() => changeScope('')}>Show all history</button>}</div> : (
        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
          <div className="min-w-0 overflow-hidden rounded-xl border bg-muted/10">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-card px-3 py-2">
              <div className="flex items-center gap-3"><div role="group" aria-label="History display" className="flex gap-1">{(['graph', 'steps'] as const).map((mode) => <button type="button" key={mode} aria-pressed={presentation === mode} onClick={() => setPresentation(mode)} className={`rounded px-2 py-1 text-xs font-medium ${presentation === mode ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'}`}>{mode === 'graph' ? 'Graph' : 'Steps'}</button>)}</div><p className="text-xs text-muted-foreground">{graph.nodes.length} steps · {layout.edges.length} connections</p></div>
              <div className={presentation === 'graph' ? "flex items-center gap-1" : "hidden"}><button type="button" aria-label="Zoom out" className={`${BUTTON} px-2 py-1.5`} disabled={zoom <= 0.25} onClick={() => setZoom((value) => Math.max(0.25, Math.round((value - 0.15) * 100) / 100))}><Minus className="h-4 w-4" /></button><span className="w-12 text-center text-xs" aria-live="polite">{Math.round(zoom * 100)}%</span><button type="button" aria-label="Zoom in" className={`${BUTTON} px-2 py-1.5`} disabled={zoom >= 1.6} onClick={() => setZoom((value) => Math.min(1.6, Math.round((value + 0.15) * 100) / 100))}><Plus className="h-4 w-4" /></button><button type="button" onClick={fit} className={`${BUTTON} px-2 py-1.5`}><Maximize2 className="h-4 w-4" /> Fit</button><button type="button" onClick={() => setZoom(1)} className={`${BUTTON} px-2 py-1.5`}>100%</button></div>
            </div>
            {presentation === 'steps' && <div>
              <div className="flex flex-wrap gap-2 border-b p-3"><input aria-label="Search history steps" value={stepSearch} onChange={(event) => setStepSearch(event.target.value)} placeholder="Search steps or notes…" className="min-w-40 flex-1 rounded-md border bg-background px-3 py-2 text-sm" /><select aria-label="Filter history step type" value={stepKind} onChange={(event) => setStepKind(event.target.value)} className="rounded-md border bg-background px-3 py-2 text-sm"><option value="">All step types</option>{Object.entries(KIND_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></div>
              <p className="border-b px-3 py-2 text-xs text-muted-foreground">Creation order · oldest first. Steps without recorded times appear last.</p>
              <ol aria-label="History steps in creation order" className="max-h-[590px] divide-y overflow-auto">{steps.map((node) => <li key={node.id}><button type="button" onClick={() => selectNode(node.id)} aria-pressed={selectedId === node.id} className={`w-full space-y-1 px-4 py-3 text-left hover:bg-muted/50 ${selectedId === node.id ? 'bg-primary/10 ring-2 ring-inset ring-primary' : ''}`}><span className="flex flex-wrap items-center justify-between gap-2"><span className="break-words text-sm font-semibold">{node.label}</span><span className="text-xs text-muted-foreground">{dateLabel(node.created_at, true)}</span></span><span className="block text-xs text-muted-foreground">{KIND_LABELS[node.kind]} · {statusLabel(node.status)}{graph.context.has(node.id) ? ' · Context' : ''}</span>{organization?.items[stepOrganizationKey(node)]?.notes && <span className="line-clamp-2 block text-xs text-muted-foreground">{organization.items[stepOrganizationKey(node)].notes}</span>}</button></li>)}</ol>
              {steps.length === 0 && <p className="p-8 text-center text-sm text-muted-foreground">No steps match these filters.</p>}
            </div>}
            <div hidden={presentation !== 'graph'} ref={viewport} style={{ height: Math.min(590, Math.max(320, layout.height * zoom + 20)) }} className="overflow-auto outline-offset-[-2px] focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary" tabIndex={0} aria-label="Model history graph. Scroll to explore; tab through steps and press Enter to inspect.">
              <div style={{ width: layout.width * zoom, height: layout.height * zoom, minWidth: '100%' }}>
                <div className="relative origin-top-left" style={{ width: layout.width, height: layout.height, transform: `scale(${zoom})` }}>
                  <svg className="pointer-events-none absolute inset-0 text-slate-400 dark:text-slate-500" width={layout.width} height={layout.height} aria-hidden="true"><defs><marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z" fill="currentColor" /></marker></defs>{layout.edges.map((edge, index) => {
                    const start = layout.positions.get(edge.source)!, end = layout.positions.get(edge.target)!
                    const x1 = start.x + NODE_WIDTH, y1 = start.y + NODE_HEIGHT / 2, x2 = end.x, y2 = end.y + NODE_HEIGHT / 2
                    const selectedEdge = edge.source === selectedId || edge.target === selectedId
                    const middle = x1 + (x2 - x1) / 2
                    return <path key={`${edge.source}-${edge.target}-${index}`} d={`M ${x1} ${y1} C ${middle} ${y1}, ${middle} ${y2}, ${x2 - 5} ${y2}`} fill="none" stroke="currentColor" strokeWidth={selectedEdge ? 3 : 1.5} opacity={selectedId && !selectedEdge ? 0.3 : 0.9} markerEnd={`url(#${markerId})`} className={selectedEdge ? 'text-primary' : ''} />
                  })}</svg>
                  {layout.columns.map((column, index) => <p key={index} className="absolute text-xs font-medium text-muted-foreground" style={{ top: 22, left: 28 + index * (NODE_WIDTH + COLUMN_GAP) }}>{index === 0 ? 'Starting points' : `Dependency level ${index}`}</p>)}
                  {layout.ordered.map((node) => {
                    const position = layout.positions.get(node.id)!, active = node.id === selectedId
                    return <button key={node.id} type="button" onClick={() => selectNode(node.id)} aria-pressed={active} aria-label={`${node.label}, ${KIND_LABELS[node.kind]}, ${statusLabel(node.status)}, ${dateLabel(node.created_at, true)}`} className={`absolute overflow-hidden rounded-lg border-2 p-3 text-left shadow-sm transition-shadow focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-primary/30 ${statusClass(node)} ${active ? 'ring-4 ring-primary/35' : 'hover:shadow-md'}`} style={{ left: position.x, top: position.y, width: NODE_WIDTH, height: NODE_HEIGHT }}><span className="flex items-center justify-between gap-2 text-[11px]"><span className="font-medium text-muted-foreground">{graph.context.has(node.id) ? 'Context · ' : ''}{KIND_LABELS[node.kind]}</span><span className="truncate rounded bg-muted/70 px-1.5 py-0.5">{statusLabel(node.status)}</span></span><span className="mt-2 block overflow-hidden break-words text-sm font-semibold leading-5" style={{ display: '-webkit-box', WebkitBoxOrient: 'vertical', WebkitLineClamp: 2 }}>{node.label}</span><span className="absolute bottom-3 left-3 right-3 block truncate text-[11px] text-muted-foreground">{dateLabel(node.created_at)}</span></button>
                  })}
                </div>
              </div>
            </div>
            <p className="border-t bg-card px-3 py-2 text-xs text-muted-foreground">{presentation === 'graph' ? 'Scroll or zoom to follow branches. Timestamps show creation order; lines show recorded relationships.' : 'Select a step to inspect its outcome, organize it, or prepare a new branch. Creation order does not imply a dependency.'}</p>
          </div>
          <aside className="min-w-0 space-y-3 rounded-xl border bg-card p-4" aria-label="Selected history step">
            {selected ? <>
              <div><p className="text-xs text-muted-foreground">{KIND_LABELS[selected.kind]} · {statusLabel(selected.status)}</p><h3 className="mt-1 break-words text-lg font-semibold">{selected.label}</h3><p className="mt-1 text-xs text-muted-foreground">{dateLabel(selected.created_at, true)}</p></div>
              {selectedItem?.label && recordedLabel !== selected.label && <p className="break-words text-xs text-muted-foreground">Recorded as: {recordedLabel}</p>}
              {selectedItem?.notes && <p className="whitespace-pre-wrap break-words text-sm">{selectedItem.notes}</p>}
              {selectedItem?.experiment_ids.length ? <div className="flex flex-wrap gap-1">{organization?.experiments.filter((experiment) => selectedItem.experiment_ids.includes(experiment.id)).map((experiment) => <span key={experiment.id} className="rounded-full bg-muted px-2 py-1 text-xs">{experiment.name}</span>)}</div> : null}
              {graph.context.has(selected.id) && <p className="text-xs text-muted-foreground">Context: this ancestor is shown to explain a step assigned to the current experiment.</p>}
              <OrganizationEditor key={selected.id} itemKey={stepOrganizationKey(selected)} originalLabel={recordedLabel || selected.label} />
              {selected.missing_reason && <p className="rounded-md bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">{selected.missing_reason}</p>}
              {typeof selected.settings?.archived_at === 'string' && <p className="rounded-lg bg-muted p-3 text-sm text-muted-foreground">This run was deleted. Its settings and result summary are preserved here. Deleted artifacts must be restored or recreated before they can be used again.</p>}
              <div className="flex flex-col gap-2">
                {selected.links?.comparison ? <Link href={selected.links.comparison} className={`${BUTTON} border-primary bg-primary text-primary-foreground hover:bg-primary/90`}>Open benchmark comparison <ArrowUpRight className="h-4 w-4" /></Link> : selected.links?.resume && <Link href={selected.links.resume} className={`${BUTTON} border-primary bg-primary text-primary-foreground hover:bg-primary/90`}><GitBranch className="h-4 w-4" />{['failed', 'cancelled'].includes(selected.status) ? 'Review and retry as a new run' : 'Branch from this step'}</Link>}
                {selected.links?.run && <Link href={selected.links.run} className={BUTTON}>Inspect full results <ArrowUpRight className="h-4 w-4" /></Link>}
              </div>
              {selected.links?.comparison ? <p className="text-xs text-muted-foreground">Compare this run with the other models, review its scoring, or set up the next comparison. This step stays saved.</p> : selected.links?.resume && <p className="text-xs text-muted-foreground">Opens a prefilled form for review. Starting a branch creates a new run and preserves this step.</p>}
              {typeof selected.settings?.resume_note === 'string' && <p className="rounded-lg bg-muted p-3 text-sm text-muted-foreground">{selected.settings.resume_note}</p>}
              <OutcomeHighlights node={selected} />
              <SettingsHighlights node={selected} />
              <JsonDetails title="Full result data" value={selected.summary} empty="No outcome was recorded for this step." />
              <JsonDetails title="Settings" value={selected.settings} empty="No settings were recorded for this step." />
              <details className="rounded-lg border p-3"><summary className="cursor-pointer text-sm font-medium">Provenance</summary><dl className="mt-2 space-y-2 break-all text-xs"><div><dt className="font-medium">Recorded by</dt><dd className="text-muted-foreground">{selected.origin || 'Unknown'}</dd></div><div><dt className="font-medium">Step ID</dt><dd className="font-mono text-muted-foreground">{selected.id}</dd></div>{selected.model_ref && <div><dt className="font-medium">Model</dt><dd className="font-mono text-muted-foreground">{selected.model_ref}</dd></div>}{selected.source_model && <div><dt className="font-medium">Source model</dt><dd className="font-mono text-muted-foreground">{selected.source_model}</dd></div>}</dl></details>
              {(incoming.length > 0 || outgoing.length > 0) && <div className="space-y-3 border-t pt-3">{[['Inputs', incoming, 'source'], ['Next steps', outgoing, 'target']].map(([label, connections, direction]) => { const items = connections as typeof incoming; return items.length > 0 && <div key={label as string}><h4 className="text-xs font-semibold">{label as string}</h4><ul className="mt-1 space-y-1">{items.map((edge, index) => { const id = edge[direction as 'source' | 'target']; const next = namedNodes.find((node) => node.id === id); return <li key={`${id}-${index}`}><button onClick={() => selectNode(id, true)} className="w-full rounded px-1 py-1 text-left text-xs text-primary hover:bg-muted"><span className="block">{next?.label ?? id}</span><span className="text-[11px] text-muted-foreground">{edge.label}</span></button></li> })}</ul></div> })}</div>}
            </> : <div className="py-8 text-center"><GitBranch className="mx-auto h-7 w-7 text-muted-foreground" /><h3 className="mt-3 font-semibold">Inspect a step</h3><p className="mt-2 text-sm text-muted-foreground">Select a checkpoint or experiment to see its settings, outcomes, and connections. You can return to any available step and branch from it.</p></div>}
          </aside>
        </div>
      )}
    </div>
  )
}
