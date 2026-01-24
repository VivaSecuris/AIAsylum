import { Layout } from '@/components/layout/Layout'
import { MetricCard } from '@/components/common/MetricCard'
import { DataTable } from '@/components/common/DataTable'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { useTestRuns } from '@/lib/hooks'
import { TestRun } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import { useMemo } from 'react'
import Link from 'next/link'
import { PlayCircle, TestTube } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

export default function Dashboard() {
  const { data: testRuns = [], isLoading, error } = useTestRuns({ limit: 100 })

  const metrics = useMemo(() => {
    const total = testRuns.length
    const running = testRuns.filter((r) => r.status === 'running').length
    const completed = testRuns.filter((r) => r.status === 'completed').length
    const failed = testRuns.filter((r) => r.status === 'failed').length

    return { total, running, completed, failed }
  }, [testRuns])

  const statusData = [
    { name: 'Completed', value: metrics.completed, color: '#10b981' },
    { name: 'Running', value: metrics.running, color: '#3b82f6' },
    { name: 'Pending', value: testRuns.filter((r) => r.status === 'pending').length, color: '#f59e0b' },
    { name: 'Failed', value: metrics.failed, color: '#ef4444' },
  ]

  const recentRuns = testRuns.slice(0, 10)

  const tableColumns = [
    {
      key: 'id',
      header: 'ID',
    },
    {
      key: 'test_type',
      header: 'Type',
      render: (run: TestRun) => (
        <span className="capitalize">{run.test_type}</span>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (run: TestRun) => <StatusBadge status={run.status} />,
    },
    {
      key: 'doctor_model',
      header: 'Doctor',
      render: (run: TestRun) => `${run.doctor_provider}/${run.doctor_model}`,
    },
    {
      key: 'patient_model',
      header: 'Patient',
      render: (run: TestRun) => `${run.patient_provider}/${run.patient_model}`,
    },
    {
      key: 'created_at',
      header: 'Created',
      render: (run: TestRun) => (run.created_at ? formatDate(run.created_at) : '-'),
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (run: TestRun) => (
        <Link
          href={`/test-runs/${run.id}`}
          className="text-primary hover:underline"
        >
          View
        </Link>
      ),
    },
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

  if (error) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-2">Error loading data</h2>
            <p className="text-muted-foreground">
              {error instanceof Error ? error.message : 'An unexpected error occurred'}
            </p>
          </div>
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <div className="flex gap-3">
            <Link
              href="/create-test"
              className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              <PlayCircle className="h-4 w-4" />
              Create Test
            </Link>
            <Link
              href="/benchmarks"
              className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              <TestTube className="h-4 w-4" />
              Run Benchmark
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard title="Total Tests" value={metrics.total} />
          <MetricCard title="Running" value={metrics.running} />
          <MetricCard title="Completed" value={metrics.completed} />
          <MetricCard title="Failed" value={metrics.failed} />
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border bg-card p-6">
            <h2 className="mb-4 text-lg font-semibold">Status Overview</h2>
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

          <div className="rounded-lg border bg-card p-6">
            <h2 className="mb-4 text-lg font-semibold">Recent Test Runs</h2>
            <DataTable
              data={recentRuns}
              columns={tableColumns}
              onRowClick={(run) => window.location.href = `/test-runs/${run.id}`}
              emptyMessage="No test runs yet"
            />
          </div>
        </div>
      </div>
    </Layout>
  )
}
