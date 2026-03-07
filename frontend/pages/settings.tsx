import { useState, useEffect } from 'react'
import { Eye, EyeOff, CheckCircle2, Circle } from 'lucide-react'
import { Layout } from '@/components/layout/Layout'
import { ModelSelector } from '@/components/forms/ModelSelector'
import { getSettings, saveSettings, DEFAULT_SETTINGS, AppSettings } from '@/lib/settings'
import { toast } from '@/lib/toast'
import { apiClient } from '@/lib/api'

interface ApiKeyState {
  value: string
  show: boolean
  configured: boolean
}

const EMPTY_KEY_STATE: ApiKeyState = { value: '', show: false, configured: false }

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings>({ ...DEFAULT_SETTINGS })
  const [saved, setSaved] = useState(false)
  const [savingKeys, setSavingKeys] = useState(false)

  const [openaiKey, setOpenaiKey] = useState<ApiKeyState>(EMPTY_KEY_STATE)
  const [anthropicKey, setAnthropicKey] = useState<ApiKeyState>(EMPTY_KEY_STATE)
  const [googleKey, setGoogleKey] = useState<ApiKeyState>(EMPTY_KEY_STATE)

  useEffect(() => {
    setSettings(getSettings())
    apiClient.getApiKeysStatus().then((status) => {
      setOpenaiKey((k) => ({ ...k, configured: status.openai_configured }))
      setAnthropicKey((k) => ({ ...k, configured: status.anthropic_configured }))
      setGoogleKey((k) => ({ ...k, configured: status.google_configured }))
    }).catch(() => {
      // backend may not be running; silently ignore
    })
  }, [])

  async function handleSaveKeys() {
    setSavingKeys(true)
    try {
      const payload: Record<string, string> = {}
      if (openaiKey.value.trim()) payload.openai_api_key = openaiKey.value.trim()
      if (anthropicKey.value.trim()) payload.anthropic_api_key = anthropicKey.value.trim()
      if (googleKey.value.trim()) payload.google_api_key = googleKey.value.trim()

      if (Object.keys(payload).length === 0) {
        toast.info('No new keys entered')
        return
      }

      const result = await apiClient.updateApiKeys(payload)
      setOpenaiKey((k) => ({ ...k, value: '', configured: result.openai_configured }))
      setAnthropicKey((k) => ({ ...k, value: '', configured: result.anthropic_configured }))
      setGoogleKey((k) => ({ ...k, value: '', configured: result.google_configured }))
      toast.success('API keys saved')
    } catch {
      toast.error('Failed to save API keys')
    } finally {
      setSavingKeys(false)
    }
  }

  function handleSave() {
    saveSettings(settings)
    setSaved(true)
    toast.success('Settings saved')
    setTimeout(() => setSaved(false), 2000)
  }

  function handleReset() {
    setSettings({ ...DEFAULT_SETTINGS })
    saveSettings({ ...DEFAULT_SETTINGS })
    toast.info('Settings reset to defaults')
  }

  function update<K extends keyof AppSettings>(key: K, value: AppSettings[K]) {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <Layout>
      <div className="mx-auto max-w-3xl px-6 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="mt-2 text-muted-foreground">
            Configure default models and analysis options. These pre-fill new test runs and analysis dialogs so you
            don&apos;t have to repeat yourself.
          </p>
        </div>

        <div className="space-y-8">
          {/* API Keys */}
          <section className="rounded-lg border bg-card p-6">
            <h2 className="mb-1 text-lg font-semibold">API Keys</h2>
            <p className="mb-6 text-sm text-muted-foreground">
              Keys are stored server-side in your <code className="rounded bg-muted px-1 py-0.5 text-xs">.env</code> file
              and applied immediately. Values are never echoed back.
            </p>

            <div className="space-y-4">
              {(
                [
                  { label: 'OpenAI API Key', placeholder: 'sk-...', state: openaiKey, setState: setOpenaiKey },
                  { label: 'Anthropic API Key', placeholder: 'sk-ant-...', state: anthropicKey, setState: setAnthropicKey },
                  { label: 'Google API Key', placeholder: 'AIza...', state: googleKey, setState: setGoogleKey },
                ] as const
              ).map(({ label, placeholder, state, setState }) => (
                <div key={label}>
                  <div className="mb-1.5 flex items-center gap-2">
                    <label className="text-sm font-medium">{label}</label>
                    {state.configured ? (
                      <span className="flex items-center gap-1 text-xs text-emerald-500">
                        <CheckCircle2 className="h-3.5 w-3.5" /> Configured
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Circle className="h-3.5 w-3.5" /> Not set
                      </span>
                    )}
                  </div>
                  <div className="relative">
                    <input
                      type={state.show ? 'text' : 'password'}
                      value={state.value}
                      onChange={(e) => setState((s) => ({ ...s, value: e.target.value }))}
                      placeholder={state.configured ? '••••••••••••••••' : placeholder}
                      className="w-full rounded-md border border-input bg-background pr-10 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring font-mono"
                    />
                    <button
                      type="button"
                      onClick={() => setState((s) => ({ ...s, show: !s.show }))}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                      tabIndex={-1}
                    >
                      {state.show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-5 flex justify-end">
              <button
                type="button"
                onClick={handleSaveKeys}
                disabled={savingKeys}
                className="rounded-lg bg-primary px-5 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                {savingKeys ? 'Saving…' : 'Save API Keys'}
              </button>
            </div>
          </section>

          {/* Default Models */}
          <section className="rounded-lg border bg-card p-6">
            <h2 className="mb-1 text-lg font-semibold">Default Models</h2>
            <p className="mb-6 text-sm text-muted-foreground">
              Pre-fill the provider and model fields when creating new test runs.
            </p>

            <div className="space-y-6">
              <ModelSelector
                label="Default Doctor Model"
                provider={settings.defaultDoctorProvider}
                model={settings.defaultDoctorModel}
                onProviderChange={(v) => update('defaultDoctorProvider', v)}
                onModelChange={(v) => update('defaultDoctorModel', v)}
              />
              <ModelSelector
                label="Default Patient Model"
                provider={settings.defaultPatientProvider}
                model={settings.defaultPatientModel}
                onProviderChange={(v) => update('defaultPatientProvider', v)}
                onModelChange={(v) => update('defaultPatientModel', v)}
              />
            </div>
          </section>

          {/* Default Analysis LLM */}
          <section className="rounded-lg border bg-card p-6">
            <h2 className="mb-1 text-lg font-semibold">Default Evaluator Model</h2>
            <p className="mb-6 text-sm text-muted-foreground">
              The AI model used to evaluate conversations during analysis. Leave empty to use the doctor model
              as the evaluator.
            </p>

            <ModelSelector
              label="Evaluator Model"
              provider={settings.defaultEvaluatorProvider}
              model={settings.defaultEvaluatorModel}
              onProviderChange={(v) => update('defaultEvaluatorProvider', v)}
              onModelChange={(v) => update('defaultEvaluatorModel', v)}
            />
          </section>

          {/* Default Analysis Options */}
          <section className="rounded-lg border bg-card p-6">
            <h2 className="mb-1 text-lg font-semibold">Default Analysis Options</h2>
            <p className="mb-6 text-sm text-muted-foreground">
              These options are pre-selected whenever you configure analysis for a test run.
            </p>

            <div className="space-y-6">
              {/* CoT Detection */}
              <div className="rounded-lg border bg-muted/30 p-4">
                <h3 className="mb-4 font-medium">Chain of Thought Analysis</h3>
                <div className="space-y-4">
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.defaultEnableCotDetection}
                      onChange={(e) => update('defaultEnableCotDetection', e.target.checked)}
                      className="h-4 w-4 rounded border-gray-300"
                    />
                    <div>
                      <span className="font-medium">Enable Chain of Thought Detection</span>
                      <p className="text-xs text-muted-foreground">
                        Detect and analyze chain of thought reasoning patterns in conversations
                      </p>
                    </div>
                  </label>

                  {settings.defaultEnableCotDetection && (
                    <div>
                      <label className="mb-2 block text-sm font-medium">Default Analysis Mode</label>
                      <select
                        value={settings.defaultCotAnalysisMode}
                        onChange={(e) =>
                          update('defaultCotAnalysisMode', e.target.value as AppSettings['defaultCotAnalysisMode'])
                        }
                        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <option value="full">Full Analysis — Complete chain of thought detection</option>
                        <option value="partial">Partial Analysis — Basic pattern detection</option>
                        <option value="none">None — Disable CoT analysis</option>
                      </select>
                    </div>
                  )}
                </div>
              </div>

              {/* Factuality */}
              <div className="rounded-lg border bg-muted/30 p-4">
                <h3 className="mb-4 font-medium">Factuality Analysis</h3>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.defaultEnableFactualityCheck}
                    onChange={(e) => update('defaultEnableFactualityCheck', e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300"
                  />
                  <div>
                    <span className="font-medium">Enable Factuality / Hallucination Detection</span>
                    <p className="text-xs text-muted-foreground">
                      Verify if responses are factual and identify potential hallucinations
                    </p>
                  </div>
                </label>
              </div>

              {/* Manipulation */}
              <div className="rounded-lg border bg-muted/30 p-4">
                <h3 className="mb-4 font-medium">Manipulation Analysis</h3>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.defaultEnableManipulationAnalysis}
                    onChange={(e) => update('defaultEnableManipulationAnalysis', e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300"
                  />
                  <div>
                    <span className="font-medium">Enable Manipulation Analysis</span>
                    <p className="text-xs text-muted-foreground">
                      Analyze manipulation resistance and manipulation capability scores
                    </p>
                  </div>
                </label>
              </div>
            </div>
          </section>
        </div>

        {/* Actions */}
        <div className="mt-8 flex items-center justify-between">
          <button
            type="button"
            onClick={handleReset}
            className="rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted transition-colors"
          >
            Reset to Defaults
          </button>
          <button
            type="button"
            onClick={handleSave}
            className="rounded-lg bg-primary px-6 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            {saved ? 'Saved!' : 'Save Settings'}
          </button>
        </div>
      </div>
    </Layout>
  )
}
