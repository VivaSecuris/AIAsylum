import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/router'
import Link from 'next/link'
import { Plus, BarChart3, Server } from 'lucide-react'
import { Layout } from '@/components/layout/Layout'
import { CampaignBuilder } from '@/components/benchmarks/CampaignBuilder'
import { CampaignResults } from '@/components/benchmarks/CampaignResults'
import { activeCampaign, useBenchmarkCampaign, useBenchmarkCampaigns, useStartBenchmarkCampaign, type BenchmarkCampaignRequest } from '@/lib/benchmark-campaigns'
import { useModelCatalog } from '@/lib/model-catalog'
import { formatApiError, formatDateTime } from '@/lib/utils'

export default function BenchmarksPage() {
  const router = useRouter()
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  const id = typeof router.query.campaign === 'string' ? router.query.campaign : undefined
  const pageTop = useRef<HTMLDivElement>(null)
  useEffect(() => { pageTop.current?.scrollIntoView({ block: 'start' }) }, [id])
  const { data: list, isLoading: listLoading, error: listError, refetch: refreshList } = useBenchmarkCampaigns()
  const { data: campaign, isLoading, error, refetch: refreshCampaign } = useBenchmarkCampaign(id)
  const { data: catalog } = useModelCatalog()
  const start = useStartBenchmarkCampaign()
  const [initial, setInitial] = useState<BenchmarkCampaignRequest>()
  const [formKey, setFormKey] = useState(0)
  const newComparison = (repeat = false) => {
    setInitial(repeat && campaign ? { name: campaign.name, models: campaign.models, benchmarks: campaign.benchmarks, num_samples: campaign.num_samples, seed: campaign.seed, max_new_tokens: campaign.max_new_tokens, enable_cot: campaign.enable_cot ?? false, temperature: campaign.temperature ?? 0, top_p: campaign.top_p, patient_system_prompt_id: campaign.patient_system_prompt ? undefined : campaign.patient_system_prompt_id, patient_system_prompt: campaign.patient_system_prompt, patient_prompt_framing: campaign.patient_prompt_framing } : undefined)
    setFormKey((key) => key + 1)
    start.reset()
    void router.push('/benchmarks', undefined, { shallow: true })
  }

  return <Layout><div ref={pageTop} className="mx-auto max-w-6xl space-y-6">
    <header className="space-y-3"><div className="flex flex-wrap items-start justify-between gap-4"><div><h1 className="text-3xl font-bold">Benchmark comparisons</h1><p className="mt-2 max-w-3xl text-sm text-muted-foreground">Compare originals and custom models on the same questions, investigate differences, and keep the evidence for your next change.</p></div><Link href="/compare?view=charts" className="inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted"><BarChart3 className="h-4 w-4" /> All visualizations</Link></div>{catalog?.location && <p className="flex items-center gap-1.5 text-xs text-muted-foreground"><Server className="h-3.5 w-3.5" /> Connected server: {catalog.location}</p>}</header>
    <div className="flex flex-wrap items-end gap-3 rounded-xl border bg-card p-4"><label className="min-w-64 flex-1 text-sm font-medium">Saved comparison<select aria-label="Saved comparison" value={id ?? ''} onChange={(event) => event.target.value ? void router.push({ pathname: '/benchmarks', query: { campaign: event.target.value } }, undefined, { shallow: true }) : newComparison()} className="mt-1 w-full rounded-md border bg-background px-3 py-2 font-normal"><option value="">New comparison</option>{id && !list?.campaigns.some((item) => item.id === id) && <option value={id}>{campaign?.name ?? 'Loading comparison…'}</option>}{list?.campaigns.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.status} · {formatDateTime(item.created_at)}</option>)}</select></label>{id && <button type="button" onClick={() => newComparison()} className="inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted"><Plus className="h-4 w-4" /> New comparison</button>}</div>
    {listLoading && <p role="status" className="text-sm text-muted-foreground">Loading saved comparisons…</p>}
    {(error || listError) && <div role="alert" className="space-y-2 rounded-lg border border-destructive/30 p-4 text-sm"><p className="text-destructive">{error && campaign ? 'Connection interrupted. Showing the last received results; remote runs can continue while disconnected.' : formatApiError(error || listError, 'Could not load saved comparisons')}</p><button type="button" className="rounded border px-3 py-1.5 hover:bg-muted" onClick={() => { void refreshList(); if (id) void refreshCampaign() }}>Reconnect</button></div>}
    {!id && !!list?.campaigns.length && <section aria-label="Recent benchmark comparisons" className="grid gap-3 md:grid-cols-3">{list.campaigns.slice(0, 3).map((item) => <Link key={item.id} href={{ pathname: '/benchmarks', query: { campaign: item.id } }} className="rounded-lg border bg-card p-4 hover:border-primary"><p className="break-words text-sm font-medium">{item.name}</p><p className="mt-1 text-xs text-muted-foreground">{item.models.length} models · {item.completed} / {item.total} complete</p><p className="mt-2 text-xs font-medium text-primary">{activeCampaign(item) ? 'View running comparison →' : 'View saved results →'}</p></Link>)}</section>}
    {id ? <>{isLoading && <p role="status" className="rounded-xl border p-8 text-center text-muted-foreground">Loading benchmark results…</p>}{campaign && <CampaignResults key={campaign.id} campaign={campaign} onRepeat={() => newComparison(true)} />}</> : mounted && <CampaignBuilder key={formKey} initial={initial} pending={start.isPending} error={start.error ? formatApiError(start.error, 'Could not start the comparison') : null} onStart={(request) => start.mutate(request, { onSuccess: (created) => { void router.push({ pathname: '/benchmarks', query: { campaign: created.id } }, undefined, { shallow: true }) } })} />}
  </div></Layout>
}
