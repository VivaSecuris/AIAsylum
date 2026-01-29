import { useState, useMemo } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Layout } from '@/components/layout/Layout'
import { TestRunTable } from '@/components/test-runs/TestRunTable'
import { TestRunFilters } from '@/components/test-runs/TestRunFilters'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useTestRuns, useDeleteTestRun } from '@/lib/hooks'
import { TestRun } from '@/lib/api'
import Link from 'next/link'
import { Plus, Trash2 } from 'lucide-react'
import { toast } from '@/lib/toast'

export default function TestRunsPage() {
  const [filters, setFilters] = useState<{
    test_type?: string
    status?: string
    search?: string
  }>({})
  const queryClient = useQueryClient()
  const { data: testRuns = [], isLoading, error } = useTestRuns({ limit: 1000 })
  const deleteTestRun = useDeleteTestRun()

  const filteredRuns = useMemo(() => {
    let filtered = [...testRuns]

    if (filters.test_type) {
      filtered = filtered.filter((r) => r.test_type === filters.test_type)
    }

    if (filters.status) {
      filtered = filtered.filter((r) => r.status === filters.status)
    }

    if (filters.search) {
      const searchLower = filters.search.toLowerCase()
      filtered = filtered.filter(
        (r) =>
          r.doctor_model.toLowerCase().includes(searchLower) ||
          r.patient_model.toLowerCase().includes(searchLower) ||
          r.doctor_provider.toLowerCase().includes(searchLower) ||
          r.patient_provider.toLowerCase().includes(searchLower)
      )
    }

    return filtered
  }, [testRuns, filters])

  const handleDelete = async (id: number) => {
    console.log('🔴 Delete button clicked for test run:', id)
    if (!confirm('Are you sure you want to delete this test run? This action cannot be undone.')) {
      console.log('Delete cancelled by user')
      return
    }
    
    console.log('✅ User confirmed deletion, calling mutation for test run:', id)
    try {
      console.log('📡 Calling deleteTestRun.mutateAsync with id:', id)
      const result = await deleteTestRun.mutateAsync(id)
      console.log('✅ Delete mutation successful, result:', result)
      toast.success('Test run deleted successfully')
      // Force refetch - invalidate and refetch
      await queryClient.invalidateQueries({ queryKey: ['test-runs'] })
      await queryClient.refetchQueries({ queryKey: ['test-runs'] })
      console.log('✅ Queries invalidated and refetched')
    } catch (error: any) {
      console.error('❌ Delete failed for test run:', id)
      console.error('Error object:', error)
      console.error('Error response:', error?.response)
      console.error('Error data:', error?.response?.data)
      const errorMessage = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Failed to delete test run'
      console.error('Error message:', errorMessage)
      toast.error(errorMessage)
      alert(`Delete failed: ${errorMessage}`) // Also show alert for debugging
    }
  }

  const handleDeleteAllFailed = async () => {
    const failedRuns = filteredRuns.filter((r) => r.status === 'failed')
    if (failedRuns.length === 0) {
      toast.info('No failed test runs to delete')
      return
    }
    if (
      confirm(
        `Are you sure you want to delete ${failedRuns.length} failed test run(s)? This action cannot be undone.`
      )
    ) {
      try {
        // Delete all failed runs sequentially
        for (const run of failedRuns) {
          await deleteTestRun.mutateAsync(run.id)
        }
        toast.success(`Successfully deleted ${failedRuns.length} failed test run(s)`)
      } catch (error: any) {
        const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to delete some test runs'
        toast.error(errorMessage)
        console.error('Failed to delete test runs:', error)
      }
    }
  }

  const failedCount = filteredRuns.filter((r) => r.status === 'failed').length

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
    const errorMessage = error instanceof Error ? error.message : String(error)
    const isNetworkError = errorMessage.includes('connect') || errorMessage.includes('Network Error')
    
    return (
      <Layout>
        <div className="rounded-lg border bg-card p-8 text-center max-w-md mx-auto">
          <h2 className="text-xl font-semibold mb-2">Error loading test runs</h2>
          <p className="text-muted-foreground mb-4">
            {errorMessage}
          </p>
          {isNetworkError && (
            <div className="text-sm text-muted-foreground space-y-2">
              <p>Please check:</p>
              <ul className="list-disc list-inside space-y-1 text-left">
                <li>The backend API server is running</li>
                <li>The API URL is correct (check browser console)</li>
                <li>No firewall is blocking the connection</li>
              </ul>
            </div>
          )}
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Test Runs</h1>
          <Link
            href="/create-test"
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Plus className="h-4 w-4" />
            Create Test
          </Link>
        </div>

        <TestRunFilters onFilterChange={setFilters} />

        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Showing {filteredRuns.length} of {testRuns.length} test runs
            {failedCount > 0 && (
              <span className="ml-2 text-destructive">({failedCount} failed)</span>
            )}
          </div>
          {failedCount > 0 && (
            <button
              onClick={handleDeleteAllFailed}
              disabled={deleteTestRun.isPending}
              className="flex items-center gap-2 rounded-lg border border-destructive px-4 py-2 text-sm font-medium text-destructive hover:bg-destructive/10 disabled:opacity-50"
            >
              <Trash2 className="h-4 w-4" />
              {deleteTestRun.isPending ? 'Deleting...' : `Delete All Failed (${failedCount})`}
            </button>
          )}
        </div>

        <TestRunTable testRuns={filteredRuns} onDelete={handleDelete} />
      </div>
    </Layout>
  )
}
