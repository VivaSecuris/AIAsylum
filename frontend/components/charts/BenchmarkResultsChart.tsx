import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line } from 'recharts'

interface BenchmarkResultsChartProps {
  data: Array<{ name: string; score: number }>
  type?: 'bar' | 'line'
  /** Pixel height, or `'100%'` when the parent supplies a sized box. */
  height?: number | `${number}%`
}

export function BenchmarkResultsChart({ data, type = 'bar', height = 300 }: BenchmarkResultsChartProps) {
  if (type === 'line') {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis domain={[0, 100]} />
          <Tooltip />
          <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis domain={[0, 100]} />
        <Tooltip />
        <Bar dataKey="score" fill="#3b82f6" />
      </BarChart>
    </ResponsiveContainer>
  )
}
