import { useEffect } from 'react'
import { useRouter } from 'next/router'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

export default function BenchmarksPage() {
  const router = useRouter()

  useEffect(() => {
    // Redirect to create-test page with benchmark type
    router.replace('/create-test?type=benchmark')
  }, [router])

  return (
    <Layout>
      <div className="flex h-full items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    </Layout>
  )
}
