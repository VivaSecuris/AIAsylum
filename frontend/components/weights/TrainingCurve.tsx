import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

// One record per optimizer step ({step, loss, lr, ...}) or per evaluation
// ({step, eval_loss}), as weights/lora.py logs them. Extra keys are ignored.
export interface TrainingPoint {
  step: number
  loss?: number | null
  lr?: number | null
  eval_loss?: number | null
  kl?: number | null
  ce?: number | null
  phase?: string
}

export function TrainingCurve({ history }: { history: TrainingPoint[] }) {
  const byStep = new Map<number, { step: number; loss?: number; eval_loss?: number; kl?: number }>()
  for (const point of history ?? []) {
    if (typeof point?.step !== 'number') continue
    const row = byStep.get(point.step) ?? { step: point.step }
    if (typeof point.loss === 'number') row.loss = point.loss
    if (typeof point.eval_loss === 'number') row.eval_loss = point.eval_loss
    if (typeof point.kl === 'number') row.kl = point.kl
    byStep.set(point.step, row)
  }
  const data = Array.from(byStep.values()).sort((a, b) => a.step - b.step)
  if (!data.some((row) => row.loss != null || row.eval_loss != null)) return null
  const hasKl = data.some((row) => row.kl != null)

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <h2 className="text-lg font-semibold">Training curve</h2>
      <p className="mb-4 mt-1 text-sm text-muted-foreground">
        Training loss per optimizer step, with the held-out evaluation loss where it was measured.
        A training loss that keeps falling while the eval loss rises is memorisation, not learning;
        the eval loss is the number to read.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} width={48} />
          <Tooltip formatter={(value: any) => (typeof value === 'number' ? value.toFixed(4) : value)} />
          <Legend />
          <Line type="monotone" dataKey="loss" name="train loss" dot={false} stroke="#2563eb" strokeWidth={1.5} connectNulls />
          <Line type="monotone" dataKey="eval_loss" name="eval loss" stroke="#dc2626" strokeWidth={2} connectNulls dot={{ r: 3 }} />
          {hasKl && <Line type="monotone" dataKey="kl" name="KL to teacher" dot={false} stroke="#7c3aed" strokeDasharray="4 2" connectNulls />}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
