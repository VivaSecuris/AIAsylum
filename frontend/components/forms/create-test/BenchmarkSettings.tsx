import Link from 'next/link'

import type { FormState } from '@/lib/create-test-config'
import { useBenchmarks } from '@/lib/hooks'
import { INPUT } from './ui'

export function BenchmarkSettings({ state, update }: {
  state: FormState
  update: (patch: Partial<FormState>) => void
}) {
  const { data } = useBenchmarks()
  const benchmarks: any[] = data?.benchmarks ?? []
  const categories: any[] = data?.categories?.length ? data.categories : [{ id: '_all', title: 'Benchmarks' }]
  const inCategory = (id: string) => (id === '_all' ? benchmarks : benchmarks.filter((b) => b.category === id))

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <span className="text-sm font-medium">Benchmark</span>
        <div className="space-y-3">
          {categories.map((category) => {
            const rows = inCategory(category.id)
            if (rows.length === 0) return null
            return (
              <div key={category.id}>
                {category.id !== '_all' && <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{category.title}</h4>}
                <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                  {rows.map((benchmark) => {
                    const unavailable = benchmark.runnable === false
                    return (
                      <button
                        key={benchmark.name}
                        type="button"
                        disabled={unavailable}
                        aria-pressed={state.benchmark === benchmark.name}
                        onClick={() => update({ benchmark: benchmark.name })}
                        className={`rounded border p-3 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                          state.benchmark === benchmark.name ? 'border-primary bg-primary/10' : 'bg-muted/30 hover:bg-muted/50'
                        }`}
                      >
                        <span className="font-medium">{benchmark.title ?? benchmark.name}</span>
                        <span className="mt-1 block text-xs text-muted-foreground">{benchmark.description}</span>
                        {unavailable && <span className="mt-1 block text-xs text-destructive">{benchmark.unavailable_reason}</span>}
                      </button>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        <label className="text-xs font-medium">
          Number of samples
          <input type="number" min="1" max="10000" value={state.numSamples} placeholder="100"
            onChange={(e) => update({ numSamples: e.target.value })} className={INPUT} />
        </label>
        <label className="text-xs font-medium">
          Question selection seed
          <input type="number" min="0" max="4294967295" value={state.benchmarkSeed} placeholder="0"
            onChange={(e) => update({ benchmarkSeed: e.target.value })} className={INPUT} />
        </label>

      </div>
      <p className="text-xs text-muted-foreground">
        Benchmarks use independent questions and the standard answer prompt. The patient’s generation settings apply; blank temperature uses 0. The seed picks the questions and controls generation where the provider supports it.{' '}
        <Link href="/benchmarks" className="text-primary underline">Compare several models in one benchmark campaign.</Link>
      </p>
    </div>
  )
}
