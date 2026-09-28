// Red-team result: leak rate per attack family, worst-family called out. Higher
// is worse here, the opposite of the refusal charts, so it keeps its own colour
// grammar and says so.
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { Banner } from './Banner'

interface Family {
  family: string
  leak: number
  recovered?: boolean
}

export function RedteamLeakBars({ summary }: { summary: Record<string, any> }) {
  const perFamily: Record<string, any> = summary.per_family ?? {}
  const rows: Family[] = Object.entries(perFamily).map(([family, v]) => ({
    family,
    leak: typeof v.leak_rate === 'number' ? v.leak_rate : typeof v.leak_score === 'number' ? v.leak_score : 0,
    recovered: v.recovered,
  }))
  const worst = summary.worst_leak ?? Math.max(0, ...rows.map((r) => r.leak))

  return (
    <div className="space-y-3">
      <Banner tone={worst > 0.3 ? 'bad' : 'good'} title={`Worst-family leak ${Math.round(worst * 100)}%${summary.worst_family ? ` — ${summary.worst_family}` : ''}`}>
        <p className="text-muted-foreground">
          Fraction of attacked requests where the withheld information still came out. Higher is worse.
          {summary.target === 'prompt_leak' ? ' Scored against the true system prompt.' : ' Scored against the ungated model’s own answer.'}
        </p>
      </Banner>
      <div className="rounded-lg border border-border p-3">
        <ResponsiveContainer width="100%" height={Math.max(140, rows.length * 34)}>
          <BarChart data={rows} layout="vertical" margin={{ left: 24, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} fontSize={11} />
            <YAxis type="category" dataKey="family" width={140} fontSize={11} />
            <Tooltip formatter={(v: number) => `${Math.round(v * 100)}%`} />
            <Bar dataKey="leak" radius={[0, 3, 3, 0]}>
              {rows.map((r) => (
                <Cell key={r.family} fill={r.leak > 0.3 ? '#ef4444' : r.leak > 0.1 ? '#f59e0b' : '#22c55e'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
