import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { ArrowRight, BarChart3, GitBranch, Brain, Scissors, RefreshCw } from 'lucide-react'
import { activeCampaign, benchmarkName, shortModelName, useCancelBenchmarkCampaign, useRescoreBenchmarkCampaign, type BenchmarkCampaign, type BenchmarkCampaignRun } from '@/lib/benchmark-campaigns'
import { formatDateTime, formatApiError } from '@/lib/utils'

const statusLabel: Record<string, string> = { pending: 'Queued', running: 'Running', completed: 'Completed', partial: 'Completed with gaps', failed: 'Failed', cancelled: 'Cancelled' }
const runStatus = (run: BenchmarkCampaignRun) => statusLabel[run.status] ?? run.status
const hasScore = (run: BenchmarkCampaignRun) => run.status === 'completed' && run.score != null && Number.isFinite(run.score)
const percent = (value: number) => `${(value * 100).toFixed(1)}%`
const linkClass = 'inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted'

function RunEvidence({ run }: { run: BenchmarkCampaignRun }) {
  const entries = [
    ['Dataset & sampled questions', run.provenance],
    ['Model checkpoint', run.model_provenance],
    ['Generation settings', run.generation],
    ['Scoring protocol', run.scoring],
  ] as const
  if (!entries.some(([, value]) => value != null)) return null
  return <details className="mt-2 text-xs"><summary className="cursor-pointer text-muted-foreground">Dataset, checkpoint & settings</summary><div className="mt-2 max-w-lg space-y-3">{entries.map(([title, value]) => <div key={title}><h4 className="font-medium">{title}</h4>{value == null ? <p className="mt-1 text-muted-foreground">Not yet recorded.</p> : <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded bg-muted p-3">{typeof value === 'string' ? value : JSON.stringify(value, null, 2)}</pre>}</div>)}</div></details>
}

function Progress({ run }: { run: BenchmarkCampaignRun }) {
  const progress = run.progress
  if (!progress || run.status !== 'running') return null
  const done = progress.completed ?? progress.current ?? progress.completed_samples ?? progress.samples_completed
  const total = progress.total ?? progress.total_samples ?? run.requested_samples
  return typeof done === 'number' && typeof total === 'number' ? <span className="block text-xs text-muted-foreground">{done} / {total} questions</span> : null
}

export function CampaignResults({ campaign, onRepeat }: { campaign: BenchmarkCampaign; onRepeat: () => void }) {
  const cancel = useCancelBenchmarkCampaign()
  const rescore = useRescoreBenchmarkCampaign()
  const router = useRouter()
  const [focusModel, setFocusModel] = useState(campaign.models[0] ?? '')
  const [showCancel, setShowCancel] = useState(false)
  const running = activeCampaign(campaign)
  const current = campaign.runs.find((run) => run.status === 'running')
  const warnings = campaign.warnings.filter((warning) => !warning.startsWith('Small-sample screening'))
  const finished = campaign.runs.filter((run) => ['completed', 'failed', 'cancelled'].includes(run.status)).length
  const successful = campaign.runs.filter(hasScore).length
  const progressPercent = campaign.total ? (finished / campaign.total) * 100 : 0
  const model = campaign.models.includes(focusModel) ? focusModel : campaign.models[0] ?? ''

  return <div className="space-y-6">
    <section className="space-y-4 rounded-xl border bg-card p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4"><div><div className="flex flex-wrap items-center gap-2"><h2 className="text-xl font-semibold">{campaign.name}</h2><span role="status" className={`rounded-full px-2.5 py-1 text-xs font-medium ${running ? 'bg-blue-500/10 text-blue-600 dark:text-blue-300' : campaign.status === 'completed' ? 'bg-green-500/10 text-green-600 dark:text-green-300' : 'bg-amber-500/10 text-amber-700 dark:text-amber-300'}`}>{statusLabel[campaign.status] ?? campaign.status}</span></div><p className="mt-2 text-xs text-muted-foreground">Created {formatDateTime(campaign.created_at)} · Saved comparison</p></div>
        {running ? <button type="button" onClick={() => setShowCancel(true)} disabled={cancel.isPending} className={linkClass}>Cancel comparison</button> : <button type="button" onClick={onRepeat} className={linkClass}><RefreshCw className="h-4 w-4" /> Retry or change models</button>}
      </div>
      {showCancel && running && <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 text-sm"><p>Cancel the current run and the remaining queue? Completed results stay available.</p><div className="mt-3 flex gap-2"><button type="button" disabled={cancel.isPending} onClick={() => cancel.mutate(campaign.id, { onSuccess: () => setShowCancel(false) })} className="rounded border px-3 py-1.5 hover:bg-muted disabled:opacity-50">{cancel.isPending ? 'Cancelling…' : 'Cancel remaining runs'}</button><button type="button" disabled={cancel.isPending} onClick={() => setShowCancel(false)} className="rounded border px-3 py-1.5 hover:bg-muted">Keep running</button></div></div>}
      {cancel.error && <p role="alert" className="text-sm text-destructive">{formatApiError(cancel.error, 'Could not cancel the comparison')}</p>}
      {campaign.source_campaign_id && <p className="rounded-lg bg-muted p-3 text-sm">Re-scored from saved answers; model responses and generation settings are unchanged. <Link href={{ pathname: '/benchmarks', query: { campaign: campaign.source_campaign_id } }} className="text-primary underline">View original comparison</Link></p>}
      {!running && campaign.scoring_outdated && successful === campaign.total && <div className="space-y-2 rounded-lg border border-amber-500/30 p-3 text-sm"><p>A newer answer scorer is available. Create a corrected comparison using these saved answers. The original scores stay available, and models do not need to run again.</p><button type="button" disabled={rescore.isPending} className={linkClass} onClick={() => rescore.mutate(campaign.id, { onSuccess: (updated) => { void router.push({ pathname: '/benchmarks', query: { campaign: updated.id } }, undefined, { shallow: true }) } })}>{rescore.isPending ? 'Re-scoring saved answers…' : 'Re-score saved answers'}</button>{rescore.error && <p role="alert" className="text-destructive">{formatApiError(rescore.error, 'Could not re-score the saved answers')}</p>}</div>}
      <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4"><div><dt className="text-muted-foreground">Models & checks</dt><dd className="mt-1 font-medium">{campaign.models.length} models × {campaign.benchmarks.length} checks</dd></div><div><dt className="text-muted-foreground">Questions per check</dt><dd className="mt-1 font-medium">{campaign.num_samples}</dd></div><div><dt className="text-muted-foreground">Sampling seed</dt><dd className="mt-1 font-medium">{campaign.seed}</dd></div><div><dt className="text-muted-foreground">Maximum answer tokens</dt><dd className="mt-1 font-medium">{campaign.max_new_tokens}</dd></div></dl>
      <div><div className="mb-2 flex justify-between text-sm"><span>{finished} / {campaign.total} runs finished</span><span className="text-muted-foreground">{successful} scored</span></div><div role="progressbar" aria-label="Benchmark runs finished" aria-valuemin={0} aria-valuemax={campaign.total} aria-valuenow={finished} className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-primary transition-all" style={{ width: `${Math.min(100, progressPercent)}%` }} /></div></div>
      {running && <p className="text-xs text-muted-foreground">The server processes one benchmark at a time. This page updates automatically; your queue keeps running if you leave.</p>}
      <p className="text-xs text-muted-foreground">Small sample screening, not a published leaderboard. Scores use generated answers and exact task scoring; results from other evaluation protocols may differ.</p>
      {campaign.comparable && <p className="text-sm text-green-700 dark:text-green-300">All runs completed with matching questions and settings within each check.</p>}
      {!campaign.comparable && !running && <p className="text-sm text-amber-700 dark:text-amber-300">This comparison has gaps or unverified matching samples. Review the run details before drawing conclusions.</p>}
      {current && <div className="rounded-lg bg-primary/5 p-3 text-sm"><span className="font-medium">Currently running: {shortModelName(current.model)} · {benchmarkName(current.benchmark)}</span><Progress run={current} /></div>}
      {!!warnings.length && <ul className="space-y-1 rounded-lg border border-amber-500/30 p-3 text-sm text-amber-700 dark:text-amber-300">{warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>}
      <nav aria-label="Comparison sections" className="flex flex-wrap gap-2"><a href="#comparison-scores" className={linkClass}>Compare scores</a><a href="#comparison-queue" className={linkClass}>Queue & answers</a><a href="#comparison-investigate" className={linkClass}>Investigate a model</a></nav>
    </section>

    <section id="comparison-scores" aria-labelledby="benchmark-results-title" className="space-y-3"><div><h2 id="benchmark-results-title" className="text-xl font-semibold">Compare results by check</h2><p className="mt-1 text-sm text-muted-foreground">Each bar shows the fraction of questions answered correctly. Failed and queued runs have no score.</p></div>
      <div className="grid gap-4 xl:grid-cols-2">{campaign.benchmarks.map((benchmark) => <article key={benchmark} className="rounded-xl border bg-card p-5"><h3 className="font-semibold">{benchmarkName(benchmark)}</h3><div className="mt-4 space-y-4">{campaign.models.map((ref) => {
        const run = campaign.runs.find((item) => item.model === ref && item.benchmark === benchmark)
        const scored = run && hasScore(run)
        return <div key={ref}>
          <div className="mb-1.5 flex items-baseline justify-between gap-3 text-sm"><span title={ref} className="min-w-0 break-words">{shortModelName(ref)}</span><span className="shrink-0 font-medium">{scored ? percent(run.score!) : run ? runStatus(run) : 'Queued'}</span></div>
          <div className="h-3 overflow-hidden rounded-full bg-muted" aria-label={scored ? `${shortModelName(ref)}: ${percent(run.score!)}` : `${shortModelName(ref)}: no score`}><div className="h-full rounded-full bg-primary" style={{ width: `${scored ? Math.max(0, Math.min(100, run.score! * 100)) : 0}%` }} /></div>
          {scored && <p className="mt-1 text-xs text-muted-foreground">{run.num_correct ?? '—'} correct / {run.num_samples ?? '—'} scored{run.num_samples !== run.requested_samples ? ` (${run.requested_samples} requested)` : ''}</p>}
          {scored && (!!run.truncated_count || !!run.invalid_answer_count) && <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">{run.truncated_count ?? 0} answers hit the token limit · {run.invalid_answer_count ?? 0} had no valid final answer</p>}
        </div>
      })}</div></article>)}</div>
    </section>

    <section id="comparison-queue" className="overflow-hidden rounded-xl border bg-card"><div className="border-b p-5"><h2 className="text-lg font-semibold">Run queue & evidence</h2><p className="mt-1 text-sm text-muted-foreground">Open a run to inspect its questions and answers. Evidence includes recorded dataset and sample details.</p></div><div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="bg-muted/40 text-xs text-muted-foreground"><tr><th className="px-4 py-3">Model</th><th className="px-4 py-3">Check</th><th className="px-4 py-3">Status</th><th className="px-4 py-3">Accuracy</th><th className="px-4 py-3">Evidence</th></tr></thead><tbody className="divide-y">{campaign.runs.map((run) => <tr key={`${run.model}:${run.benchmark}`} className="align-top"><td className="max-w-64 px-4 py-3"><span className="break-words font-medium" title={run.model}>{shortModelName(run.model)}</span><Link href={{ pathname: '/compare', query: { view: 'history', model: run.model } }} className="mt-1 block text-xs text-primary hover:underline">Model history</Link></td><td className="whitespace-nowrap px-4 py-3">{benchmarkName(run.benchmark)}</td><td className="min-w-32 px-4 py-3"><span className={run.status === 'failed' ? 'text-destructive' : ''}>{runStatus(run)}</span><Progress run={run} />{run.error && <details className="mt-2 max-w-sm text-xs text-destructive"><summary className="cursor-pointer">Failure details</summary><p className="mt-1 whitespace-pre-wrap break-words">{run.error}</p></details>}</td><td className="whitespace-nowrap px-4 py-3">{hasScore(run) ? <><span className="font-medium">{percent(run.score!)}</span><span className="block text-xs text-muted-foreground">{run.num_correct ?? '—'} / {run.num_samples ?? '—'}</span></> : <span className="text-muted-foreground">—</span>}</td><td className="px-4 py-3">{run.id ? <Link href={`/test-runs/${run.id}`} className="whitespace-nowrap text-primary hover:underline">Open run #{run.id} →</Link> : <span className="text-xs text-muted-foreground">Run not started</span>}<RunEvidence run={run} /></td></tr>)}</tbody></table></div></section>

    <section id="comparison-investigate" className="space-y-4 rounded-xl border bg-primary/5 p-5"><div><h2 className="text-lg font-semibold">Investigate, change, and compare again</h2><p className="mt-1 text-sm text-muted-foreground">Choose a model, inspect its internal behavior or history, then create a new comparison after a change. This result stays saved.</p></div><label className="block max-w-xl text-sm font-medium">Model to investigate<select value={model} onChange={(event) => setFocusModel(event.target.value)} className="mt-1 w-full rounded border bg-background px-3 py-2">{campaign.models.map((ref) => <option key={ref} value={ref}>{shortModelName(ref)}</option>)}</select></label><div className="flex flex-wrap gap-2"><Link href={{ pathname: '/compare', query: { view: 'history', model } }} className={linkClass}><GitBranch className="h-4 w-4" /> Trace model history</Link><Link href={{ pathname: '/interp', query: { mode: 'single', model_a: model } }} className={linkClass}><Brain className="h-4 w-4" /> Inspect internals</Link><Link href={{ pathname: '/weights', query: { source_model: model } }} className={linkClass}><Scissors className="h-4 w-4" /> Explore a modification</Link><Link href="/compare?view=charts" className={linkClass}><BarChart3 className="h-4 w-4" /> All visualizations</Link><button type="button" onClick={onRepeat} className={linkClass}>Set up next comparison <ArrowRight className="h-4 w-4" /></button></div></section>
  </div>
}
