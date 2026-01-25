import { useState, useMemo } from 'react'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { DataTable } from '@/components/common/DataTable'
import { useTestRuns } from '@/lib/hooks'
import { TestRun } from '@/lib/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, ComposedChart } from 'recharts'

export default function ComparePage() {
  const { data: testRuns = [], isLoading } = useTestRuns({ limit: 1000 })
  const [selectedTestType, setSelectedTestType] = useState<string>('')

  // Fetch assessments for all completed test runs
  const completedRunIds = useMemo(() => 
    testRuns
      .filter(run => run.status === 'completed' && (!selectedTestType || run.test_type === selectedTestType))
      .map(run => run.id),
    [testRuns, selectedTestType]
  )

  // We'll aggregate assessment data in the modelStats calculation
  const modelStats = useMemo(() => {
    const stats: Record<string, {
      provider: string
      model: string
      testCount: number
      completedCount: number
      avgScore: number
      testTypes: Record<string, number>
      statusCounts: Record<string, number>
      scores: number[]
    }> = {}

    testRuns.forEach((run) => {
      if (selectedTestType && run.test_type !== selectedTestType) return

      const key = `${run.patient_provider}:${run.patient_model}`
      if (!stats[key]) {
        stats[key] = {
          provider: run.patient_provider,
          model: run.patient_model,
          testCount: 0,
          completedCount: 0,
          avgScore: 0,
          testTypes: {},
          statusCounts: {},
          scores: [],
        }
      }
      stats[key].testCount++
      stats[key].statusCounts[run.status] = (stats[key].statusCounts[run.status] || 0) + 1
      if (run.status === 'completed') {
        stats[key].completedCount++
      }
      if (run.test_type) {
        stats[key].testTypes[run.test_type] = (stats[key].testTypes[run.test_type] || 0) + 1
      }
    })

    return Object.values(stats).sort((a, b) => b.testCount - a.testCount)
  }, [testRuns, selectedTestType])

  // Calculate average scores per model (this would ideally come from assessments, but for now we'll show test counts)
  const chartData = modelStats.slice(0, 10).map((stat) => ({
    name: stat.model.length > 15 ? stat.model.substring(0, 15) + '...' : stat.model,
    fullName: stat.model,
    'Test Count': stat.testCount,
    'Completed': stat.completedCount,
    'Pending': stat.statusCounts.pending || 0,
    'Failed': stat.statusCounts.failed || 0,
  }))

  const testTypeData = useMemo(() => {
    const typeCounts: Record<string, number> = {}
    testRuns.forEach((run) => {
      if (selectedTestType && run.test_type !== selectedTestType) return
      if (run.status !== 'completed') return
      const type = run.test_type || 'unknown'
      typeCounts[type] = (typeCounts[type] || 0) + 1
    })
    return Object.entries(typeCounts).map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }))
  }, [testRuns, selectedTestType])

  const statusData = useMemo(() => {
    const statusCounts: Record<string, number> = {}
    testRuns.forEach((run) => {
      if (selectedTestType && run.test_type !== selectedTestType) return
      statusCounts[run.status] = (statusCounts[run.status] || 0) + 1
    })
    return Object.entries(statusCounts).map(([name, value]) => ({ 
      name: name.charAt(0).toUpperCase() + name.slice(1), 
      value,
      color: name === 'completed' ? '#10b981' : name === 'running' ? '#3b82f6' : name === 'failed' ? '#ef4444' : '#6b7280'
    }))
  }, [testRuns, selectedTestType])

  const modelComparisonData = modelStats.slice(0, 5).map((stat) => ({
    model: stat.model.length > 12 ? stat.model.substring(0, 12) + '...' : stat.model,
    fullName: stat.model,
    'Conversation': stat.testTypes.conversation || 0,
    'Scenario': stat.testTypes.scenario || 0,
    'Adversarial': stat.testTypes.adversarial || 0,
  }))

  const columns = [
    { key: 'model', header: 'Model' },
    { key: 'provider', header: 'Provider' },
    { key: 'testCount', header: 'Total Tests' },
    { key: 'completedCount', header: 'Completed' },
    { 
      key: 'testTypes', 
      header: 'Test Types',
      render: (stat: typeof modelStats[0]) => (
        <div className="flex gap-1 flex-wrap">
          {Object.entries(stat.testTypes).map(([type, count]) => (
            <span key={type} className="text-xs bg-muted px-2 py-1 rounded">
              {type}: {count}
            </span>
          ))}
        </div>
      )
    },
  ]

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

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
          <div>
            <h1 className="text-3xl font-bold">Models</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Compare model performance across test runs and analysis results
            </p>
          </div>
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

        {/* Summary Cards */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">Total Models</h3>
            <p className="text-2xl font-bold">{modelStats.length}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">Total Test Runs</h3>
            <p className="text-2xl font-bold">{testRuns.filter(r => !selectedTestType || r.test_type === selectedTestType).length}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">Completed Runs</h3>
            <p className="text-2xl font-bold text-green-600">
              {testRuns.filter(r => r.status === 'completed' && (!selectedTestType || r.test_type === selectedTestType)).length}
            </p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-1">Active Models</h3>
            <p className="text-2xl font-bold">
              {modelStats.filter(m => m.completedCount > 0).length}
            </p>
          </div>
        </div>

        {/* Charts Row 1 */}
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
            <h2 className="text-lg font-semibold mb-4">Test Run Status Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Charts Row 2 */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Test Type Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={testTypeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Model Status Breakdown</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Completed" stackId="a" fill="#10b981" />
                <Bar dataKey="Pending" stackId="a" fill="#f59e0b" />
                <Bar dataKey="Failed" stackId="a" fill="#ef4444" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Charts Row 3 */}
        {modelComparisonData.length > 0 && (
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Top Models by Test Type</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Conversation" stackId="a" fill="#3b82f6" />
                <Bar dataKey="Scenario" stackId="a" fill="#10b981" />
                <Bar dataKey="Adversarial" stackId="a" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Model Statistics Table */}
        <div className="rounded-lg border bg-card p-6">
          <h2 className="text-lg font-semibold mb-4">Model Statistics</h2>
          <DataTable
            data={modelStats}
            columns={columns}
            emptyMessage="No model data available"
          />
        </div>
      </div>
    </Layout>
  )
}
