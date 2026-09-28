import { useEffect, useState } from 'react'
import { useRouter } from 'next/router'

import { Layout } from '@/components/layout/Layout'
import { CreateTestForm } from '@/components/forms/create-test/CreateTestForm'
import { WorkflowEntry } from '@/components/benchmarks/WorkflowEntry'

const RESTORE_KEYS = ['model', 'provider', 'type', 'test_config', 'doctor_model', 'doctor_provider', 'lineage_parent', 'promptId']

export default function CreateTestPage() {
  const router = useRouter()
  // The form reads Settings (browser storage) and the URL once, when it mounts, so
  // it renders only on the client after the router has the query: no hydration
  // mismatch and no defaults flashing before the link's values arrive.
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  const restoreKey = JSON.stringify(RESTORE_KEYS.map((key) => router.query[key] ?? null))

  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Create New Test Run</h1>
        <WorkflowEntry compact />
        {typeof router.query.lineage_parent === 'string' && (
          <p className="rounded-lg border bg-primary/5 p-4 text-sm">
            Review the recorded model and test settings before starting a new run. The previous result is preserved in the history graph.
          </p>
        )}
        {mounted && router.isReady ? (
          <CreateTestForm key={restoreKey} query={router.query} />
        ) : (
          <div className="h-40 animate-pulse rounded-lg border bg-muted/30" aria-label="Loading form" />
        )}
      </div>
    </Layout>
  )
}
