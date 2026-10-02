import Link from 'next/link'
import { Layers, PlayCircle, TestTube } from 'lucide-react'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { WorkflowEntry } from '@/components/benchmarks/WorkflowEntry'
import { DashboardGrid } from '@/components/dashboard/DashboardGrid'
import { useDashboardData } from '@/components/dashboard/useDashboardData'
import { formatApiError } from '@/lib/utils'

export default function Dashboard() {
  const { data, isLive, isLoading, initialError, refreshError, retry } = useDashboardData()
  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <div className="flex flex-wrap gap-3">
            <Link href="/create-test" className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90">
              <PlayCircle className="h-4 w-4" /> Create Test
            </Link>
            <Link href="/suite" className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted">
              <Layers className="h-4 w-4" /> Create Suite
            </Link>
            <Link href="/benchmarks" className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted">
              <TestTube className="h-4 w-4" /> Compare benchmarks
            </Link>
          </div>
        </div>
        <WorkflowEntry compact />
        {isLoading ? (
          <div className="flex min-h-[240px] items-center justify-center"><LoadingSpinner size="lg" /></div>
        ) : initialError ? (
          <div role="alert" className="rounded-lg border p-6">
            <h2 className="font-semibold">Error loading dashboard data</h2>
            <p className="mt-2 text-sm text-muted-foreground">{formatApiError(initialError)}</p>
            <button type="button" onClick={retry} className="mt-3 text-sm text-primary hover:underline">Try again</button>
          </div>
        ) : (
          <>
            {refreshError && (
              <div role="status" className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm">
                Some data could not be refreshed. Showing the latest available results.
                <button type="button" onClick={retry} className="ml-2 text-primary hover:underline">Retry</button>
              </div>
            )}
            <DashboardGrid data={data} isLive={isLive} />
          </>
        )}
      </div>
    </Layout>
  )
}
