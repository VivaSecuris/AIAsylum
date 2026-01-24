import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts'

interface ModelComparisonChartProps {
  data: Array<{ name: string; [key: string]: string | number }>
  dataKeys: string[]
}

export function ModelComparisonChart({ data, dataKeys }: ModelComparisonChartProps) {
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        {dataKeys.map((key, index) => (
          <Bar key={key} dataKey={key} fill={colors[index % colors.length]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
