import { useState, useEffect } from 'react'

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
  ollama: ['llama2', 'llama3', 'llama3.2', 'mistral', 'mixtral', 'codellama'],
}

export function ModelSelector({
  label,
  provider,
  model,
  onProviderChange,
  onModelChange,
}: ModelSelectorProps) {
  const [availableModels, setAvailableModels] = useState<string[]>([])

  useEffect(() => {
    if (provider) {
      setAvailableModels(MODELS_BY_PROVIDER[provider] || [])
      // Reset model if not available for new provider
      if (model && !MODELS_BY_PROVIDER[provider]?.includes(model)) {
        onModelChange('')
      }
    }
  }, [provider, model, onModelChange])

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
            disabled={!provider}
            className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <option value="">Select model</option>
            {availableModels.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}
