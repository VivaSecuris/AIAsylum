import { useEffect } from 'react'
import { useRouter } from 'next/router'
import Link from 'next/link'
import { Layout } from '@/components/layout/Layout'

export default function BenchmarkDetailPage() {
  const router = useRouter()

  useEffect(() => {
    if (router.isReady) void router.replace('/benchmarks')
  }, [router.isReady, router.replace])

  return (
    <Layout>
      <div className="space-y-3 rounded-lg border bg-card p-6">
        <h1 className="text-xl font-semibold">Benchmark comparisons</h1>
        <p role="status" className="text-sm text-muted-foreground">
          Opening saved comparisons and benchmark setup…
        </p>
        <Link href="/benchmarks" className="inline-block text-sm text-primary hover:underline">
          Open benchmark comparisons
        </Link>
      </div>
    </Layout>
  )
}
