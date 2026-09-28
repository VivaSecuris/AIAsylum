// Hallucination-neuron result: how many neurons were selected and, crucially,
// whether the detector clears its baselines. The honest "no signal" case is the
// common one on small models, so it is shown plainly rather than buried.
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { MetricCard } from '@/components/common/MetricCard'
import { usableHNeuronSignal } from '@/lib/weight-result-evidence'
import { Banner } from './Banner'

export function HNeuronResults({ summary }: { summary: Record<string, any> }) {
  const bars = [
    { k: 'AUROC', v: summary.auroc ?? 0 },
    { k: 'shuffled null', v: summary.null_auroc_p95 ?? 0 },
    { k: 'length baseline', v: summary.surface_auroc ?? 0 },
  ]
  const usable = usableHNeuronSignal(summary)
  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-3">
        <MetricCard title="Neurons selected" value={String(summary.n_selected ?? '—')} subtitle={typeof summary.fraction === 'number' ? `${(summary.fraction * 100).toFixed(3)}% of ${summary.n_total}` : undefined} />
        <MetricCard title="Detector AUROC" value={typeof summary.auroc === 'number' ? summary.auroc.toFixed(3) : '—'} subtitle={`null ${typeof summary.null_auroc_p95 === 'number' ? summary.null_auroc_p95.toFixed(3) : '—'}`} />
        <MetricCard title="Labelled" value={`${summary.n_incorrect ?? '?'} wrong`} subtitle={`${summary.n_correct ?? '?'} correct`} />
      </div>
      <Banner tone={usable ? 'good' : 'warn'} title={usable ? 'The detector clears both baselines.' : 'Usable signal not established.'}>
        <p className="text-muted-foreground">
          {usable
            ? 'AUROC exceeds both the shuffled-label null and the answer-length baseline by at least 0.05. Test a down-scale separately to measure whether these neurons affect hallucination.'
            : 'A usable detector needs selected neurons and AUROC at least 0.05 above both finite baselines. One or more checks failed or were not recorded; this result does not justify a down-scale.'}
        </p>
      </Banner>
      <div className="rounded-lg border border-border p-3">
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={bars} margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" fontSize={11} />
            <YAxis domain={[0, 1]} fontSize={11} />
            <Tooltip formatter={(v: number) => v.toFixed(3)} />
            <ReferenceLine y={0.5} stroke="#9ca3af" strokeDasharray="4 2" />
            <Bar dataKey="v" radius={[3, 3, 0, 0]}>
              {bars.map((b, i) => (
                <Cell key={b.k} fill={i === 0 ? (usable ? '#22c55e' : '#f59e0b') : '#9ca3af'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
