import { useState } from 'react'
import { ModelSelector } from './ModelSelector'
import { TestRunRequest } from '@/lib/api'
import { useCreateTestRun } from '@/lib/hooks'
import { useRouter } from 'next/router'
import { toast } from '@/lib/toast'

export function CreateTestForm() {
  const router = useRouter()
  const createTestRun = useCreateTestRun()

  const [formData, setFormData] = useState<TestRunRequest>({
    doctor_provider: '',
    doctor_model: '',
    patient_provider: '',
    patient_model: '',
    test_type: 'conversation',
    test_config: {},
  })

  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const result = await createTestRun.mutateAsync(formData)
      router.push(`/test-runs/${result.id}`)
    } catch (error) {
      console.error('Failed to create test run:', error)
      toast.error('Failed to create test run. Please check your configuration.')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="rounded-lg border bg-card p-6 space-y-6">
        <h2 className="text-xl font-semibold">Test Configuration</h2>

        <ModelSelector
          label="Doctor Model"
          provider={formData.doctor_provider}
          model={formData.doctor_model}
          onProviderChange={(provider) =>
            setFormData({ ...formData, doctor_provider: provider })
          }
          onModelChange={(model) =>
            setFormData({ ...formData, doctor_model: model })
          }
        />

        <ModelSelector
          label="Patient Model"
          provider={formData.patient_provider}
          model={formData.patient_model}
          onProviderChange={(provider) =>
            setFormData({ ...formData, patient_provider: provider })
          }
          onModelChange={(model) =>
            setFormData({ ...formData, patient_model: model })
          }
        />

        <div className="space-y-2">
          <label className="text-sm font-medium">Test Type</label>
          <div className="flex gap-4">
            {['conversation', 'scenario', 'adversarial'].map((type) => (
              <label key={type} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="test_type"
                  value={type}
                  checked={formData.test_type === type}
                  onChange={(e) =>
                    setFormData({ ...formData, test_type: e.target.value })
                  }
                  className="rounded"
                />
                <span className="capitalize">{type}</span>
              </label>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-primary hover:underline"
        >
          {showAdvanced ? 'Hide' : 'Show'} Advanced Options
        </button>

        {showAdvanced && (
          <div className="space-y-4 rounded border bg-muted/50 p-4">
            {formData.test_type === 'conversation' && (
              <div>
                <label className="text-sm font-medium">Max Turns</label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={formData.test_config?.max_turns || 10}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      test_config: {
                        ...formData.test_config,
                        max_turns: parseInt(e.target.value),
                      },
                    })
                  }
                  className="mt-1 w-full rounded border px-3 py-2 text-sm"
                />
              </div>
            )}
            {formData.test_type === 'adversarial' && (
              <div>
                <label className="text-sm font-medium">Jailbreak Techniques</label>
                <select
                  multiple
                  value={formData.test_config?.jailbreak_techniques || []}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      test_config: {
                        ...formData.test_config,
                        jailbreak_techniques: Array.from(
                          e.target.selectedOptions,
                          (option) => option.value
                        ),
                      },
                    })
                  }
                  className="mt-1 w-full rounded border px-3 py-2 text-sm"
                >
                  <option value="dan">DAN (Do Anything Now)</option>
                  <option value="jailbreak">Jailbreak Prompts</option>
                  <option value="roleplay">Roleplay</option>
                  <option value="hypothetical">Hypothetical Scenarios</option>
                </select>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex justify-end gap-3">
        <button
          type="button"
          onClick={() => router.back()}
          className="rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={
            createTestRun.isPending ||
            !formData.doctor_provider ||
            !formData.doctor_model ||
            !formData.patient_provider ||
            !formData.patient_model
          }
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {createTestRun.isPending ? 'Creating...' : 'Create Test Run'}
        </button>
      </div>
    </form>
  )
}
