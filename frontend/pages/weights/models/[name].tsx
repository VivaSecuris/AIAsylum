import { useRouter } from 'next/router'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { ManifestCard, ProvenanceCard } from '@/components/weights/ManifestCard'
import { ModelChat } from '@/components/weights/ModelChat'
import { apiClient } from '@/lib/api'
import { chatErrorMessage } from '@/lib/model-chat'

export default function EditedModelPage() {
  const router = useRouter()
  const name = typeof router.query.name === 'string' ? router.query.name : ''
  const { data: model, error, isLoading } = useQuery({ queryKey: ['edited-model', name], queryFn: () => apiClient.getEditedModel(name), enabled: router.isReady && !!name, retry: false })
  const action = 'rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted'
  return <Layout><div className="space-y-6">
    <Link href={{ pathname: '/compare', query: { view: 'models' } }} className="text-sm text-primary hover:underline">← Model library</Link>
    <div><h1 className="break-words text-2xl font-bold">{name || 'Custom checkpoint'}</h1><p className="mt-1 text-sm text-muted-foreground">Inspect the saved edit, chat with its weights, or start a recorded test.</p></div>
    {isLoading && <LoadingSpinner />}
    {error && <div role="alert" className="rounded border border-destructive/40 p-4 text-sm"><p>{chatErrorMessage(error)}</p><p className="mt-2 text-muted-foreground">This checkpoint may have been deleted or its files may be unavailable. Its recorded history remains in the model library.</p></div>}
    {model && <>
      <div className="flex flex-wrap gap-2">
        <Link href={{ pathname: '/compare', query: { view: 'history', model: model.path } }} className={action}>View full history</Link>
        <Link href={{ pathname: '/create-test', query: { provider: 'transformers', model: model.path } }} className={action}>Evaluate checkpoint</Link>
        <Link href={{ pathname: '/interp', query: { mode: 'single', model_a: model.path } }} className={action}>Analyze activations</Link>
        {model.manifest.source_model && <Link href={{ pathname: '/weights', query: { kind: 'compare', source_model: model.manifest.source_model, modified_model: model.path } }} className={action}>Compare behavior with original</Link>}
      </div>
      <ModelChat key={model.path} name={model.name} />
      <div className="grid items-start gap-6 xl:grid-cols-2"><ManifestCard manifest={model.manifest} /><ProvenanceCard detail={model} /></div>
    </>}
  </div></Layout>
}
