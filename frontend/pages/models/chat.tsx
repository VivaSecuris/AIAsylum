import { useEffect, useState } from 'react'
import { useRouter } from 'next/router'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { ModelSelector } from '@/components/forms/ModelSelector'
import { ModelChat } from '@/components/weights/ModelChat'
import { apiClient } from '@/lib/api'
import { getSettings } from '@/lib/settings'
import { modelChatUrl, supportsWeightActions } from '@/lib/model-chat'
import { canUseModel, useModelCatalog } from '@/lib/model-catalog'

export default function ModelWorkspacePage() {
  const router = useRouter()
  const [selection, setSelection] = useState<{ provider: string; model: string } | null>(null)
  const [provider, setProvider] = useState('')
  const [model, setModel] = useState('')
  const { data: providers, error: providerError } = useQuery({ queryKey: ['providers'], queryFn: () => apiClient.listProviders() })
  const { data: catalog } = useModelCatalog()
  useEffect(() => {
    if (!router.isReady) return
    const saved = getSettings()
    const suppliedProvider = typeof router.query.provider === 'string' ? router.query.provider : ''
    const selected = {
      provider: suppliedProvider || saved.defaultPatientProvider,
      model: typeof router.query.model === 'string' ? router.query.model : suppliedProvider ? '' : saved.defaultPatientModel,
    }
    setSelection(selected); setProvider(selected.provider); setModel(selected.model)
  }, [router.isReady, router.query.provider, router.query.model])

  const info = providers?.find((entry) => entry.name === selection?.provider || entry.aliases.includes(selection?.provider || ''))
  const entry = supportsWeightActions(selection?.provider || '') ? catalog?.models.find((entry) => entry.model_ref === selection?.model) : undefined
  const unavailable = providerError ? 'Provider availability could not be checked. Reload this page to try again.'
    : !providers ? 'Checking provider availability…'
    : !info ? 'This provider is not supported by the connected server. Choose an available provider above.'
    : !info.available ? info.unavailable_reason || 'This provider is unavailable on the connected server.'
    : !info.configured && info.requires_api_key ? 'Configure this provider’s API key in Settings before chatting.'
    : entry && !canUseModel(entry) ? entry.reason || 'This checkpoint is unavailable on the connected server.' : undefined
  const action = 'rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted'

  return <Layout><div className="space-y-6">
    <Link href="/models" className="text-sm text-primary hover:underline">← All models</Link>
    <div><h1 className="text-2xl font-bold">Model workspace</h1><p className="mt-1 text-sm text-muted-foreground">Chat with a model, adjust its instructions, and carry the setup into tests or comparisons.</p></div>
    {!selection ? <LoadingSpinner /> : <>
      <details key={selection.model || 'choose'} open={!selection.model} className="rounded-lg border bg-card p-4"><summary className="cursor-pointer text-sm font-medium">{selection.model ? 'Change model' : 'Choose a model'}</summary>
      <form className="mt-3 space-y-3" onSubmit={(event) => { event.preventDefault(); if (provider && model.trim()) void router.push(modelChatUrl(provider, model.trim())) }}>
        <ModelSelector label="Choose a model" provider={provider} model={model} onProviderChange={(value) => { setProvider(value); setModel('') }} onModelChange={setModel} />
        <button type="submit" disabled={!provider || !model.trim() || (provider === selection.provider && model.trim() === selection.model)} className={`${action} disabled:opacity-50`}>Open model</button>
      </form>
      </details>
      {selection.model && <>
        <div className="space-y-2"><h2 className="break-words text-xl font-semibold">{entry?.name || selection.model}</h2><p className="break-all text-sm text-muted-foreground">{selection.provider} / {selection.model}</p>
          <div className="flex flex-wrap gap-2">
            <Link href={{ pathname: '/compare', query: { view: 'history', model: selection.model } }} className={action}>View history</Link>
            {entry?.kind === 'custom' && <Link href={`/weights/models/${encodeURIComponent(entry.name)}`} className={action}>Checkpoint provenance</Link>}
            {supportsWeightActions(selection.provider) && <Link href={{ pathname: '/interp', query: { mode: 'single', model_a: selection.model } }} className={action}>Analyze activations</Link>}
            {info?.requires_api_key && !info.configured && <Link href="/settings" className={action}>Provider settings</Link>}
          </div>
        </div>
        <ModelChat provider={selection.provider} model={selection.model} name={entry?.name || selection.model} sourceModel={entry?.source_model} disabledReason={unavailable} initialAdjustmentId={typeof router.query.adjustment_id === 'string' ? router.query.adjustment_id : undefined} />
      </>}
      {!selection.model && <p className="rounded border border-dashed p-8 text-center text-sm text-muted-foreground">Choose a provider and model to begin.</p>}
    </>}
  </div></Layout>
}
