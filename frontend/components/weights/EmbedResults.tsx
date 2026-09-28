// Embedding-cartography results: alignment retrieval vs its null, or the
// extraction spectrum and recovered dimension. One component, three shapes,
// switched on the run kind.
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { MetricCard } from '@/components/common/MetricCard'
import type { WeightRunKind } from '@/lib/api'
import { Banner } from './Banner'

const pct = (v: unknown) => (typeof v === 'number' ? `${Math.round(v * 100)}%` : '—')

export function EmbedResults({ kind, summary }: { kind: WeightRunKind; summary: Record<string, any> }) {
  if (kind === 'embed_extract') {
    const spectrum: number[] = (summary.spectrum ?? []).slice(0, 48)
    const data = spectrum.map((s, i) => ({ i: i + 1, s }))
    return (
      <div className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-3">
          <MetricCard title="Recovered hidden dim" value={String(summary.recovered_dim ?? '—')} subtitle={summary.true_dim ? `true ${summary.true_dim}` : undefined} />
          <MetricCard title="Subspace overlap" value={typeof summary.subspace_overlap === 'number' ? summary.subspace_overlap.toFixed(3) : '—'} subtitle="1.0 = exact" />
          <MetricCard title="Queries" value={String(summary.n_queries ?? '—')} subtitle={`vocab seen ${summary.vocab_seen ?? '—'}`} />
        </div>
        <div className="rounded-lg border border-border p-3">
          <p className="mb-2 text-xs text-muted-foreground">Singular-value spectrum — the knee marks the hidden dimension.</p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="i" fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip />
              {summary.recovered_dim ? <ReferenceLine x={summary.recovered_dim} stroke="#22c55e" strokeDasharray="4 2" label={{ value: `d=${summary.recovered_dim}`, fontSize: 11 }} /> : null}
              <Line type="monotone" dataKey="s" stroke="#3b82f6" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    )
  }

  // embed_align / embed_recon
  const retr = summary.retrieval ?? {}
  const nul = summary.null_shuffled ?? {}
  const bars = [
    { k: 'P@1', aligned: retr['1'] ?? 0, null: nul['1'] ?? 0 },
    { k: 'P@5', aligned: retr['5'] ?? 0, null: nul['5'] ?? 0 },
  ]
  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-3">
        <MetricCard title="Held-out P@1" value={pct(retr['1'])} subtitle={`shuffled null ${pct(nul['1'])}`} />
        <MetricCard title="Relative agreement" value={typeof summary.relative_agreement === 'number' ? summary.relative_agreement.toFixed(3) : '—'} subtitle={typeof summary.relative_null === 'number' ? `null ${summary.relative_null.toFixed(3)}` : 'map-free'} />
        <MetricCard title="Anchors" value={String(summary.n_anchors ?? '—')} subtitle={summary.extra?.recovered_dim ? `recovered dim ${summary.extra.recovered_dim}` : summary.method} />
      </div>
      <div className="rounded-lg border border-border p-3">
        <p className="mb-2 text-xs text-muted-foreground">
          Retrieval over the whole target vocabulary vs. a shuffled-anchor null. The gap is the real alignment.
        </p>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={bars} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" fontSize={11} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={11} />
            <Tooltip formatter={(v: number) => `${Math.round(v * 100)}%`} />
            <Bar dataKey="aligned" name="aligned" fill="#22c55e" radius={[3, 3, 0, 0]} />
            <Bar dataKey="null" name="shuffled null" fill="#9ca3af" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      {(retr['1'] ?? 0) <= (nul['1'] ?? 0) + 0.05 && (
        <Banner tone="warn" title="No better than chance.">
          <p className="text-muted-foreground">The aligned retrieval does not beat the shuffled-anchor null — these spaces do not share recoverable structure (or there are too few anchors).</p>
        </Banner>
      )}
    </div>
  )
}
