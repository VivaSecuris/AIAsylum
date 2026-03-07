import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '@/lib/api'

interface ModelSelectorProps {
  label: string
  provider: string
  model: string
  onProviderChange: (provider: string) => void
  onModelChange: (model: string) => void
}

const PROVIDERS = ['openai', 'anthropic', 'google', 'ollama']

const MODELS_BY_PROVIDER: Record<string, string[]> = {
  openai: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'],
  anthropic: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-2', 'claude-instant'],
  google: ['gemini-pro', 'gemini-pro-vision', 'palm-2'],
  ollama: [], // Fetched from API (installed models only)
}

export function ModelSelector({
  label,
  provider,
  model,
  onProviderChange,
  onModelChange,
}: ModelSelectorProps) {
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [ollamaLoading, setOllamaLoading] = useState(false)
  const [ollamaPullName, setOllamaPullName] = useState('')
  const [ollamaPulling, setOllamaPulling] = useState(false)
  const [ollamaError, setOllamaError] = useState<string | null>(null)

  const fetchOllamaModels = useCallback(async () => {
    setOllamaLoading(true)
    setOllamaError(null)
    try {
      const raw = await apiClient.listOllamaModels()
      // API returns string[]; handle wrapped { models: [...] } defensively
      const list = Array.isArray(raw)
        ? raw
        : (raw && typeof raw === 'object' && Array.isArray((raw as { models?: string[] }).models))
          ? (raw as { models: string[] }).models
          : []
      setAvailableModels(list)
      // Do NOT clear selection when list loads/refreshes - that was wiping the user's
      // model choice. Only clear when switching provider (handled in useEffect below).
    } catch (e: any) {
      setAvailableModels([])
      const detail = e?.response?.data?.detail
      setOllamaError(
        typeof detail === 'string' ? detail : (e?.message || 'Could not list Ollama models. Is Ollama running?')
      )
    } finally {
      setOllamaLoading(false)
    }
  }, [])

  useEffect(() => {
    if (provider === 'ollama') {
      fetchOllamaModels()
    } else if (provider) {
      setAvailableModels(MODELS_BY_PROVIDER[provider] || [])
      if (model && !MODELS_BY_PROVIDER[provider]?.includes(model)) {
        onModelChange('')
      }
      setOllamaError(null)
    }
  }, [provider])

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">{label}</label>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-muted-foreground">Provider</label>
          <select
            value={provider}
            onChange={(e) => onProviderChange(e.target.value)}
            className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            <option value="">Select provider</option>
            {PROVIDERS.map((p) => (
              <option key={p} value={p}>
                {p.charAt(0).toUpperCase() + p.slice(1)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Model</label>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={!provider || (provider === 'ollama' && ollamaLoading)}
            className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <option value="">
              {provider === 'ollama' && ollamaLoading ? 'Loading…' : 'Select model'}
            </option>
            {availableModels.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          {provider === 'ollama' && ollamaError && (
            <p className="mt-1 text-xs text-destructive">{ollamaError}</p>
          )}
          {provider === 'ollama' && !ollamaLoading && !ollamaError && availableModels.length === 0 && (
            <p className="mt-1 text-xs text-muted-foreground">No models installed. Pull one below or ensure Ollama is running.</p>
          )}
          {provider === 'ollama' && !ollamaLoading && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <input
                type="text"
                placeholder="Model name (e.g. llama3.2)"
                value={ollamaPullName}
                onChange={(e) => setOllamaPullName(e.target.value)}
                className="flex-1 min-w-[120px] rounded-md border border-input bg-background px-2 py-1.5 text-sm text-foreground"
              />
              <button
                type="button"
                disabled={!ollamaPullName.trim() || ollamaPulling}
                onClick={async () => {
                  const name = ollamaPullName.trim()
                  if (!name) return
                  setOllamaPulling(true)
                  setOllamaError(null)
                  try {
                    await apiClient.pullOllamaModel(name)
                    setOllamaPullName('')
                    await fetchOllamaModels()
                  } catch (e: any) {
                    setOllamaError(e?.response?.data?.detail || e?.message || 'Pull failed')
                  } finally {
                    setOllamaPulling(false)
                  }
                }}
                className="rounded-md border border-input bg-background px-2 py-1.5 text-sm text-foreground hover:bg-muted disabled:opacity-50"
              >
                {ollamaPulling ? 'Downloading…' : 'Download model'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
