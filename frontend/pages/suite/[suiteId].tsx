import { useEffect } from 'react'
import { useRouter } from 'next/router'
import { useQueryClient } from '@tanstack/react-query'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useSuite, useSuiteProgress, useSuiteRuns, useDeleteSuite } from '@/lib/hooks'
import { SuiteProgress } from '@/components/suite/SuiteProgress'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { formatDate, formatDateTime } from '@/lib/utils'
import { toast } from '@/lib/toast'
import Link from 'next/link'
import { Trash2, RefreshCw } from 'lucide-react'

export default function SuiteDetailPage() {
  const router = useRouter()
  const { suiteId } = router.query
  const id = typeof suiteId === 'string' ? parseInt(suiteId) : 0
  const queryClient = useQueryClient()

  const { data: suite, isLoading: loadingSuite } = useSuite(id)
  const { data: progress, isLoading: loadingProgress } = useSuiteProgress(
    id,
    suite?.status === 'running' || suite?.status === 'pending'
  )
  const { data: suiteRuns, isLoading: loadingRuns } = useSuiteRuns(id)
  const deleteSuite = useDeleteSuite()

  // Auto-refresh when suite is running
  useEffect(() => {
    if (suite?.status === 'running' || suite?.status === 'pending') {
      const interval = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ['suite', id] })
        queryClient.invalidateQueries({ queryKey: ['suite-progress', id] })
        queryClient.invalidateQueries({ queryKey: ['suite-runs', id] })
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [suite?.status, id, queryClient])

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this suite? This will also delete all associated test runs.')) {
      return
    }

    try {
      await deleteSuite.mutateAsync(id)
      toast.success('Suite deleted successfully')
      router.push('/suite')
    } catch (error) {
      console.error('Failed to delete suite:', error)
      toast.error('Failed to delete suite')
    }
  }

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['suite', id] })
    queryClient.invalidateQueries({ queryKey: ['suite-progress', id] })
    queryClient.invalidateQueries({ queryKey: ['suite-runs', id] })
    toast.success('Refreshed')
  }

  if (!id || id === 0) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-2">Invalid Suite ID</h2>
            <p className="text-muted-foreground">Please select a suite to view details.</p>
          </div>
        </div>
      </Layout>
    )
  }

  if (loadingSuite || loadingProgress) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </Layout>
    )
  }

  if (!suite) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-2">Suite Not Found</h2>
            <p className="text-muted-foreground">The suite you're looking for doesn't exist.</p>
          </div>
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">
              Suite #{suite.id} {suite.name && `- ${suite.name}`}
            </h1>
            <div className="mt-2 flex items-center gap-3">
              <StatusBadge status={suite.status} />
              <span className="text-sm text-muted-foreground">
                Created: {formatDate(suite.created_at)}
              </span>
              {suite.started_at && (
                <span className="text-sm text-muted-foreground">
                  Started: {formatDate(suite.started_at)}
                </span>
              )}
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleRefresh}
              className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            {suite.status !== 'running' && (
              <button
                onClick={handleDelete}
                disabled={deleteSuite.isPending}
                className="flex items-center gap-2 rounded-lg border border-destructive px-4 py-2 text-sm font-medium text-destructive hover:bg-destructive/10 disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" />
                {deleteSuite.isPending ? 'Deleting...' : 'Delete'}
              </button>
            )}
          </div>
        </div>

        {/* Progress Display */}
        {progress && (
          <div className="rounded-lg border bg-card p-6">
            <SuiteProgress
              progress={progress}
              suiteName={suite.name}
              suiteStatus={suite.status}
            />
          </div>
        )}

        {/* Test Runs List */}
        {suiteRuns && suiteRuns.length > 0 && (
          <div className="rounded-lg border bg-card">
            <div className="p-4 border-b">
              <h2 className="text-lg font-semibold">Test Runs ({suiteRuns.length})</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="px-4 py-3 text-left text-sm font-medium">ID</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Model</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Test Type</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Created</th>
                    <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {suiteRuns.map((run: any) => (
                    <tr key={run.id} className="border-b hover:bg-muted/50">
                      <td className="px-4 py-3 text-sm">#{run.id}</td>
                      <td className="px-4 py-3 text-sm">
                        {run.patient_provider}/{run.patient_model}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {run.benchmark ? (
                          <span className="capitalize">{run.benchmark}</span>
                        ) : (
                          <span className="capitalize">{run.test_type}</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={run.status} />
                      </td>
                      <td className="px-4 py-3 text-sm text-muted-foreground">
                        {formatDateTime(run.created_at)}
                      </td>
                      <td className="px-4 py-3">
                        <Link
                          href={`/test-runs/${run.id}`}
                          className="text-primary hover:underline text-sm"
                        >
                          View
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </Layout>
  )
}
