import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { MetricCard } from '@/components/common/MetricCard'

export function ProbeResults({ summary }: { summary: Record<string, any> }) {
  const data = summary.dataset ?? {}
  const number = (value: unknown) => typeof value === 'number' ? value.toFixed(3) : '—'
  return <div className="space-y-4">
    <div className="grid gap-3 sm:grid-cols-4">
      <MetricCard title="Selected layer" value={summary.best_layer ?? '—'} subtitle={summary.pooling} />
      <MetricCard title="Validation AUROC" value={number(summary.best_auroc)} subtitle="separation of harmful and benign prompts" />
      <MetricCard title="Shuffled-label null (95%)" value={number(summary.null_auroc_p95)} subtitle={summary.beats_null ? 'Above shuffled-label control' : 'Does not beat the null'} />
      <MetricCard title="Calibration error" value={number(summary.best_ece)} subtitle="lower is better" />
    </div>
    <div className="rounded-lg border bg-card p-4 text-sm">
      <h2 className="font-semibold">{summary.usable ? 'Probe meets the configured validation thresholds' : 'Probe does not meet the configured validation thresholds'}</h2>
      <p className="mt-1 text-muted-foreground">This is a descriptive detector. The same validation set selects the layer, so these scores are not an independent final test and do not establish a causal mechanism.</p>
      <p className="mt-2">{data.n_train ?? '—'} training prompts · {data.n_test ?? '—'} validation prompts · split {summary.dataset_hash ?? 'unrecorded'}</p>
      <p className="mt-1">Families excluded from training: {data.held_out_techniques?.join(', ') || 'None — no unseen-family generalization claim'}</p>
      {(data.warnings ?? []).map((warning: string) => <p key={warning} className="mt-2 text-amber-700 dark:text-amber-300">{warning}</p>)}
      {data.requested && <div className="mt-3 overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr><th>Dataset component</th><th>Requested</th><th>Available and used</th></tr></thead><tbody>{Object.entries(data.requested).map(([key, value]) => <tr key={key}><td className="py-1">{key.replaceAll('_', ' ')}</td><td>{String(value)}</td><td>{data.actual?.[key] ?? '—'}</td></tr>)}</tbody></table></div>}
    </div>
    {!!summary.layers?.length && <div className="rounded-lg border bg-card p-4">
      <h2 className="font-semibold">Detection by layer</h2>
      <p className="mb-4 text-xs text-muted-foreground">Compare separation with its shuffled-label control. Chance AUROC is 0.5.</p>
      <ResponsiveContainer width="100%" height={310}><LineChart data={summary.layers}>
        <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="layer" /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
        <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="4 4" />
        <Line dataKey="auroc" name="Validation AUROC" stroke="#2563eb" dot={false} />
        <Line dataKey="null_auroc_p95" name="Shuffled-label null (95%)" stroke="#d97706" strokeDasharray="4 4" dot={false} />
        <Line dataKey="ece" name="Calibration error" stroke="#9333ea" dot={false} />
      </LineChart></ResponsiveContainer>
    </div>}
    {!!Object.keys(summary.group_auroc ?? {}).length && <div className="rounded-lg border bg-card p-4">
      <h2 className="mb-2 font-semibold">Generalization by prompt family</h2>
      <table className="w-full text-left text-sm"><thead><tr><th>Family versus benign prompts</th><th>AUROC</th></tr></thead><tbody>{Object.entries(summary.group_auroc).map(([group, value]) => <tr key={group}><td className="py-1">{group}</td><td>{number(value)}</td></tr>)}</tbody></table>
    </div>}
  </div>
}
