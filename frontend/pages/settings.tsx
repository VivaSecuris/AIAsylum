import { useState, useEffect } from 'react'
import { Layout } from '@/components/layout/Layout'
import { ModelSelector } from '@/components/forms/ModelSelector'
import { getSettings, saveSettings, DEFAULT_SETTINGS, AppSettings } from '@/lib/settings'
import { toast } from '@/lib/toast'

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings>({ ...DEFAULT_SETTINGS })
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    setSettings(getSettings())
  }, [])

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
