import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { Layout } from '@/components/layout/Layout'
import { SuiteForm } from '@/components/forms/SuiteForm'
import { useSuites, useDeleteSuite } from '@/lib/hooks'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import Link from 'next/link'
import { formatDate } from '@/lib/utils'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { toast } from '@/lib/toast'
import { Trash2 } from 'lucide-react'

export default function SuiteIndexPage() {
  const router = useRouter()
  const [showForm, setShowForm] = useState(false)
  const { data: suites, isLoading } = useSuites()
  const deleteSuite = useDeleteSuite()
  useEffect(() => {
    if (router.isReady && (router.query.model || router.query.models || router.query.test_config)) setShowForm(true)
  }, [router.isReady, router.query.model, router.query.models, router.query.test_config])

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
          <h1 className="text-3xl font-bold">Test Suites</h1>
          <button
            onClick={() => setShowForm(!showForm)}
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            {showForm ? 'Cancel' : 'Create New Suite'}
          </button>
        </div>

        {showForm && router.isReady && (
          <div className="rounded-lg border bg-card p-6">
            <SuiteForm key={JSON.stringify(router.query)} />
          </div>
        )}

        {!showForm && (
          <div className="space-y-4">
            {suites && suites.length > 0 ? (
              <div className="rounded-lg border bg-card">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="px-4 py-3 text-left text-sm font-medium">ID</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Name</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Progress</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Created</th>
                      <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {suites.map((suite) => {
                      const progress = suite.total_runs > 0
                        ? ((suite.completed_runs / suite.total_runs) * 100).toFixed(1)
                        : '0'
                      return (
                        <tr key={suite.id} className="border-b hover:bg-muted/50">
                          <td className="px-4 py-3 text-sm">#{suite.id}</td>
                          <td className="px-4 py-3 text-sm font-medium">
                            {suite.name || 'Unnamed Suite'}
                          </td>
                          <td className="px-4 py-3">
                            <StatusBadge status={suite.status} />
                          </td>
                          <td className="px-4 py-3 text-sm">
                            {suite.completed_runs} / {suite.total_runs} ({progress}%)
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {formatDate(suite.created_at)}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-3">
                              <Link
                                href={`/suite/${suite.id}`}
                                className="text-primary hover:underline text-sm"
                              >
                                View
                              </Link>
                              <button
                                type="button"
                                onClick={() => {
                                  if (!confirm('Delete this test suite? This will also delete all associated test runs.' + (suite.status === 'running' ? ' Any running tests will be stopped.' : ''))) return
                                  deleteSuite.mutate(suite.id, {
                                    onSuccess: () => toast.success('Suite deleted'),
                                    onError: () => toast.error('Failed to delete suite'),
                                  })
                                }}
                                disabled={deleteSuite.isPending}
                                className="text-destructive hover:text-destructive/80 text-sm inline-flex items-center gap-1 disabled:opacity-50"
                                title="Delete suite"
                              >
                                <Trash2 className="h-4 w-4" />
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="rounded-lg border bg-card p-12 text-center">
                <p className="text-muted-foreground mb-4">No test suites yet</p>
                <button
                  onClick={() => setShowForm(true)}
                  className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                >
                  Create Your First Suite
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  )
}
