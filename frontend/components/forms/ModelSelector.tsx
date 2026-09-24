import { useState, useEffect, useCallback } from 'react'
import { apiClient, type ProviderInfo } from '@/lib/api'

interface ModelSelectorProps {
  label: string
  provider: string
  model: string
  onProviderChange: (provider: string) => void
  onModelChange: (model: string) => void
}

const INPUT_CLASS =
  'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50'

export function ModelSelector({
  label,
  provider,
  model,
  onProviderChange,
  onModelChange,
}: ModelSelectorProps) {
  // Providers come from the API rather than a hard-coded list: the list used to
  // drift, leaving providers registered in the backend but unreachable here.
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [providersError, setProvidersError] = useState<string | null>(null)
  const [fetchedModels, setFetchedModels] = useState<string[]>([])
  const [fetchLoading, setFetchLoading] = useState(false)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [pullName, setPullName] = useState('')
  const [pulling, setPulling] = useState(false)

  const info = providers.find(
    (p) => p.name === provider || p.aliases.includes(provider)
  )

  useEffect(() => {
    let cancelled = false
    apiClient
      .listProviders()
      .then((list) => {
        if (!cancelled) setProviders(list)
      })
      .catch((e: any) => {
        if (!cancelled) {
          setProvidersError(
            e?.response?.data?.detail || e?.message || 'Could not load providers'
          )
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const fetchModels = useCallback(async () => {
    setFetchLoading(true)
    setFetchError(null)
    try {
      const raw = await apiClient.listOllamaModels()
      const list = Array.isArray(raw)
        ? raw
        : raw && typeof raw === 'object' && Array.isArray((raw as { models?: string[] }).models)
          ? (raw as { models: string[] }).models
          : []
      setFetchedModels(list)
    } catch (e: any) {
      setFetchedModels([])
      const detail = e?.response?.data?.detail
      setFetchError(
        typeof detail === 'string'
          ? detail
          : e?.message || 'Could not list models. Is the service running?'
      )
    } finally {
      setFetchLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!info) return
    if (info.model_input === 'fetch') {
      fetchModels()
    } else {
      setFetchError(null)
      // Only clear a selection that this provider cannot offer. Free-text modes
      // keep whatever was typed.
      if (info.model_input === 'list' && model && !info.models.includes(model)) {
        onModelChange('')
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, providers.length])

  const listOptions = info?.model_input === 'fetch' ? fetchedModels : info?.models ?? []
  const isFreeText = info?.model_input === 'path' || info?.model_input === 'text'

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">{label}</label>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-muted-foreground">Provider</label>
          <select
            value={provider}
            onChange={(e) => onProviderChange(e.target.value)}
            className={INPUT_CLASS}
          >
            <option value="">Select provider</option>
            {providers.map((p) => (
              <option key={p.name} value={p.name} disabled={!p.available}>
                {p.label}
                {!p.available ? ' (unavailable)' : !p.configured ? ' (no API key)' : ''}
              </option>
            ))}
          </select>
          {providersError && (
            <p className="mt-1 text-xs text-destructive">{providersError}</p>
          )}
          {info && !info.available && info.unavailable_reason && (
            <p className="mt-1 text-xs text-destructive">{info.unavailable_reason}</p>
          )}
          {info && info.available && !info.configured && info.requires_api_key && (
            <p className="mt-1 text-xs text-destructive">
              No API key configured. Set it in Settings.
            </p>
          )}
          {info && <p className="mt-1 text-xs text-muted-foreground">{info.description}</p>}
        </div>

        <div>
          <label className="text-xs text-muted-foreground">Model</label>

          {isFreeText ? (
            <input
              type="text"
              value={model}
              placeholder={info?.placeholder ?? 'Model identifier'}
              onChange={(e) => onModelChange(e.target.value)}
              className={INPUT_CLASS}
            />
          ) : (
            <select
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
              disabled={!provider || fetchLoading}
              className={INPUT_CLASS}
            >
              <option value="">{fetchLoading ? 'Loading…' : 'Select model'}</option>
              {listOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          )}

          {info?.model_input === 'path' && (
            <p className="mt-1 text-xs text-muted-foreground">
              A local directory or a Hugging Face id. Modified models produced by{' '}
              <code>aiasylum weights ablate</code> carry their surgery manifest into each
              response.
            </p>
          )}

          {info?.model_input === 'fetch' && (
            <>
              {fetchError && <p className="mt-1 text-xs text-destructive">{fetchError}</p>}
              {!fetchLoading && !fetchError && listOptions.length === 0 && (
                <p className="mt-1 text-xs text-muted-foreground">
                  No models installed. Pull one below, or check the service is running.
                </p>
              )}
              {!fetchLoading && (
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <input
                    type="text"
                    placeholder="Model name (e.g. llama3.2)"
                    value={pullName}
                    onChange={(e) => setPullName(e.target.value)}
                    className="flex-1 min-w-[120px] rounded-md border border-input bg-background px-2 py-1.5 text-sm text-foreground"
                  />
                  <button
                    type="button"
                    disabled={!pullName.trim() || pulling}
                    onClick={async () => {
                      const name = pullName.trim()
                      if (!name) return
                      setPulling(true)
                      setFetchError(null)
                      try {
                        await apiClient.pullOllamaModel(name)
                        setPullName('')
                        await fetchModels()
                      } catch (e: any) {
                        setFetchError(
                          e?.response?.data?.detail || e?.message || 'Pull failed'
                        )
                      } finally {
                        setPulling(false)
                      }
                    }}
                    className="rounded-md border border-input bg-background px-2 py-1.5 text-sm text-foreground hover:bg-muted disabled:opacity-50"
                  >
                    {pulling ? 'Downloading…' : 'Download model'}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
