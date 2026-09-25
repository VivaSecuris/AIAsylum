import Link from 'next/link'
import { Layout } from '@/components/layout/Layout'
import { EvaluationCharts } from '@/components/models/EvaluationCharts'

export default function ModelResultsPage() {
  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <Link href="/compare?view=charts" className="mb-2 inline-block text-sm text-primary hover:underline">← Models and charts</Link>
          <h1 className="text-3xl font-bold">Evaluation results</h1>
          <p className="mt-1 text-sm text-muted-foreground">Historical behavioral assessments across your models.</p>
        </div>
        <EvaluationCharts />
      </div>
    </Layout>
  )
}
