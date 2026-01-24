import { useState, useMemo } from 'react'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { DataTable } from '@/components/common/DataTable'
import { useTestRuns } from '@/lib/hooks'
import { TestRun } from '@/lib/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts'

export default function ComparePage() {
  const { data: testRuns = [], isLoading } = useTestRuns({ limit: 1000 })
  const [selectedTestType, setSelectedTestType] = useState<string>('')

  const modelStats = useMemo(() => {
    const stats: Record<string, {
      provider: string
      model: string
      testCount: number
      completedCount: number
      avgScore: number
    }> = {}

    testRuns.forEach((run) => {
      if (selectedTestType && run.test_type !== selectedTestType) return
      if (run.status !== 'completed') return

      const key = `${run.patient_provider}:${run.patient_model}`
      if (!stats[key]) {
        stats[key] = {
          provider: run.patient_provider,
          model: run.patient_model,
          testCount: 0,
          completedCount: 0,
          avgScore: 0,
        }
      }
      stats[key].testCount++
      stats[key].completedCount++
    })

    return Object.values(stats).sort((a, b) => b.testCount - a.testCount)
  }, [testRuns, selectedTestType])

  const chartData = modelStats.slice(0, 10).map((stat) => ({
    name: stat.model,
    'Test Count': stat.testCount,
  }))

  const columns = [
    { key: 'model', header: 'Model' },
    { key: 'provider', header: 'Provider' },
    { key: 'testCount', header: 'Test Count' },
    { key: 'completedCount', header: 'Completed' },
  ]

  if (isLoading) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Model Comparison</h1>
          <select
            value={selectedTestType}
            onChange={(e) => setSelectedTestType(e.target.value)}
            className="rounded border px-3 py-2 text-sm"
          >
            <option value="">All Test Types</option>
            <option value="conversation">Conversation</option>
            <option value="scenario">Scenario</option>
            <option value="adversarial">Adversarial</option>
          </select>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Model Test Count</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Test Count" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Model Statistics</h2>
            <DataTable
              data={modelStats}
              columns={columns}
              emptyMessage="No model data available"
            />
          </div>
        </div>
      </div>
    </Layout>
  )
}
