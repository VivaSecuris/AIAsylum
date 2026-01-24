import { Layout } from '@/components/layout/Layout'
import { CreateTestForm } from '@/components/forms/CreateTestForm'
import { toast } from '@/lib/toast'

export default function CreateTestPage() {
  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Create New Test Run</h1>
        <CreateTestForm />
      </div>
    </Layout>
  )
}
