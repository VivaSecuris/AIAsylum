import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'

interface ScoreDistributionProps {
  data: Array<{ name: string; value: number }>
}

export function ScoreDistribution({ data }: ScoreDistributionProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis domain={[0, 100]} />
        <Tooltip />
        <Bar dataKey="value" fill="#3b82f6" />
      </BarChart>
    </ResponsiveContainer>
  )
}
