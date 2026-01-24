import { useRouter } from 'next/router'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { DataTable } from '@/components/common/DataTable'
import { BenchmarkResultsChart } from '@/components/charts/BenchmarkResultsChart'
import { useBenchmarks } from '@/lib/hooks'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function BenchmarkDetailPage() {
  const router = useRouter()
  const { benchmark } = router.query
  const benchmarkName = typeof benchmark === 'string' ? benchmark : ''
  const { data, isLoading } = useBenchmarks()

  const benchmarkInfo = data?.benchmarks.find((b) => b.name === benchmarkName)

  // Placeholder data - will be replaced when benchmark results API is implemented
  const placeholderResults = [
    { name: 'Category 1', score: 85.5 },
    { name: 'Category 2', score: 72.3 },
    { name: 'Category 3', score: 91.2 },
    { name: 'Category 4', score: 68.7 },
  ]

  const resultsColumns = [
    { key: 'name', header: 'Category' },
    {
      key: 'score',
      header: 'Score',
      render: (item: { score: number }) => (
        <span className="font-medium">{item.score.toFixed(1)}%</span>
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

  if (!benchmarkInfo) {
    return (
      <Layout>
        <div className="space-y-6">
          <Link
            href="/benchmarks"
            className="flex items-center gap-2 text-primary hover:underline"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Benchmarks
          </Link>
          <div className="rounded-lg border bg-card p-8 text-center">
            <h2 className="text-xl font-semibold mb-2">Benchmark not found</h2>
            <p className="text-muted-foreground">
              The benchmark "{benchmarkName}" could not be found.
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
          <div>
            <Link
              href="/benchmarks"
              className="flex items-center gap-2 text-primary hover:underline mb-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Benchmarks
            </Link>
            <h1 className="text-3xl font-bold capitalize">{benchmarkInfo.name}</h1>
            <p className="text-muted-foreground mt-1">{benchmarkInfo.description}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Results Overview</h2>
            <BenchmarkResultsChart data={placeholderResults} type="bar" />
          </div>

          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Detailed Scores</h2>
            <DataTable
              data={placeholderResults}
              columns={resultsColumns}
              emptyMessage="No results available"
            />
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <h2 className="text-lg font-semibold mb-4">Note</h2>
          <p className="text-sm text-muted-foreground">
            Benchmark results visualization is a placeholder. Full benchmark execution and
            results storage will be implemented when the benchmark API is complete.
          </p>
        </div>
      </div>
    </Layout>
  )
}
