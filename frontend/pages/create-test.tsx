import { Layout } from '@/components/layout/Layout'
import { CreateTestForm } from '@/components/forms/CreateTestForm'
import { toast } from '@/lib/toast'
import { useRouter } from 'next/router'
import { WorkflowEntry } from '@/components/benchmarks/WorkflowEntry'

export default function CreateTestPage() {
  const router = useRouter()
  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Create New Test Run</h1>
        <WorkflowEntry compact />
        {typeof router.query.lineage_parent === 'string' && <p className="rounded-lg border bg-primary/5 p-4 text-sm">Review the recorded model and test settings before starting a new run. The previous result is preserved in the history graph.</p>}
        <CreateTestForm />
      </div>
    </Layout>
  )
}
