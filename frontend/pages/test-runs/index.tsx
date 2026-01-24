import { useState, useMemo } from 'react'
import { Layout } from '@/components/layout/Layout'
import { TestRunTable } from '@/components/test-runs/TestRunTable'
import { TestRunFilters } from '@/components/test-runs/TestRunFilters'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useTestRuns } from '@/lib/hooks'
import { TestRun } from '@/lib/api'
import Link from 'next/link'
import { Plus } from 'lucide-react'

export default function TestRunsPage() {
  const [filters, setFilters] = useState<{
    test_type?: string
    status?: string
    search?: string
  }>({})
  const { data: testRuns = [], isLoading, error } = useTestRuns({ limit: 1000 })

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

  const handleDelete = (id: number) => {
    if (confirm('Are you sure you want to delete this test run?')) {
      // TODO: Implement delete API call
      console.log('Delete test run:', id)
    }
  }

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
        <div className="rounded-lg border bg-card p-8 text-center">
          <h2 className="text-xl font-semibold mb-2">Error loading test runs</h2>
          <p className="text-muted-foreground">
            {error instanceof Error ? error.message : 'An unexpected error occurred'}
          </p>
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

        <div className="text-sm text-muted-foreground">
          Showing {filteredRuns.length} of {testRuns.length} test runs
        </div>

        <TestRunTable testRuns={filteredRuns} onDelete={handleDelete} />
      </div>
    </Layout>
  )
}
