import { useState } from 'react'
import Link from 'next/link'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useBenchmarks, useRunBenchmark } from '@/lib/hooks'
import { ModelSelector } from '@/components/forms/ModelSelector'
import { Play } from 'lucide-react'
import { toast } from '@/lib/toast'

export default function BenchmarksPage() {
  const { data, isLoading } = useBenchmarks()
  const runBenchmark = useRunBenchmark()

  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('')
  const [provider, setProvider] = useState<string>('')
  const [model, setModel] = useState<string>('')
  const [numSamples, setNumSamples] = useState<number>(100)

  const handleRun = async () => {
    if (!selectedBenchmark || !provider || !model) {
      toast.error('Please select a benchmark, provider, and model')
      return
    }

    try {
      await runBenchmark.mutateAsync({
        provider,
        model,
        benchmark: selectedBenchmark,
        num_samples: numSamples,
      })
      toast.success('Benchmark started!')
    } catch (error) {
      console.error('Failed to run benchmark:', error)
      toast.error('Failed to run benchmark')
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

  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Benchmarks</h1>

        <div className="rounded-lg border bg-card p-6 space-y-6">
          <h2 className="text-xl font-semibold">Run Benchmark</h2>

          <div className="space-y-2">
            <label className="text-sm font-medium">Benchmark</label>
            <select
              value={selectedBenchmark}
              onChange={(e) => setSelectedBenchmark(e.target.value)}
              className="w-full rounded border px-3 py-2 text-sm"
            >
              <option value="">Select benchmark</option>
              {data?.benchmarks.map((benchmark) => (
                <option key={benchmark.name} value={benchmark.name}>
                  {benchmark.name} - {benchmark.description}
                </option>
              ))}
            </select>
          </div>

          <ModelSelector
            label="Model"
            provider={provider}
            model={model}
            onProviderChange={setProvider}
            onModelChange={setModel}
          />

          <div className="space-y-2">
            <label className="text-sm font-medium">Number of Samples</label>
            <input
              type="number"
              min="1"
              max="10000"
              value={numSamples}
              onChange={(e) => setNumSamples(parseInt(e.target.value))}
              className="w-full rounded border px-3 py-2 text-sm"
            />
          </div>

          <button
            onClick={handleRun}
            disabled={runBenchmark.isPending || !selectedBenchmark || !provider || !model}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            <Play className="h-4 w-4" />
            {runBenchmark.isPending ? 'Running...' : 'Run Benchmark'}
          </button>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <h2 className="text-xl font-semibold mb-4">Available Benchmarks</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data?.benchmarks.map((benchmark) => (
              <Link
                key={benchmark.name}
                href={`/benchmarks/${benchmark.name}`}
                className="rounded border p-4 hover:bg-muted/50 cursor-pointer block"
              >
                <h3 className="font-semibold capitalize">{benchmark.name}</h3>
                <p className="text-sm text-muted-foreground mt-1">{benchmark.description}</p>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  )
}
