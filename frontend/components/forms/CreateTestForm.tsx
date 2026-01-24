import { useState } from 'react'
import { ModelSelector } from './ModelSelector'
import { TestRunRequest } from '@/lib/api'
import { useCreateTestRun, usePrompts } from '@/lib/hooks'
import { useRouter } from 'next/router'
import { toast } from '@/lib/toast'
import Link from 'next/link'

export function CreateTestForm() {
  const router = useRouter()
  const createTestRun = useCreateTestRun()
  const { data: prompts = [] } = usePrompts()

  const [formData, setFormData] = useState<TestRunRequest>({
    doctor_provider: '',
    doctor_model: '',
    patient_provider: '',
    patient_model: '',
    test_type: 'multi_shot',
    test_config: {},
  })

  const [selectedPromptId, setSelectedPromptId] = useState<number | undefined>(undefined)
  const [selectedDoctorSystemPromptId, setSelectedDoctorSystemPromptId] = useState<number | undefined>(undefined)
  const [selectedPatientSystemPromptId, setSelectedPatientSystemPromptId] = useState<number | undefined>(undefined)
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  // Get system prompts separately
  const { data: doctorSystemPrompts = [] } = usePrompts({ prompt_type: 'system_prompt', target: 'doctor' })
  const { data: patientSystemPrompts = [] } = usePrompts({ prompt_type: 'system_prompt', target: 'patient' })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const testConfig = {
        ...formData.test_config,
      }
      if (selectedPromptId) {
        testConfig.prompt_id = selectedPromptId
      }
      if (selectedDoctorSystemPromptId) {
        testConfig.doctor_system_prompt_id = selectedDoctorSystemPromptId
      }
      if (selectedPatientSystemPromptId) {
        testConfig.patient_system_prompt_id = selectedPatientSystemPromptId
      }
      
      const submitData = {
        ...formData,
        test_config: testConfig,
      }
      const result = await createTestRun.mutateAsync(submitData)
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
            {[
              { value: 'one_shot', label: 'One-Shot', desc: 'Single prompt/response test' },
              { value: 'multi_shot', label: 'Multi-Shot', desc: 'Multiple sequential prompts to test context handling' },
              { value: 'conversation', label: 'Conversation', desc: 'Multi-turn conversation between doctor and patient' },
            ].map((type) => (
              <label key={type.value} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="test_type"
                  value={type.value}
                  checked={formData.test_type === type.value}
                  onChange={(e) =>
                    setFormData({ ...formData, test_type: e.target.value })
                  }
                  className="rounded"
                />
                <span className="text-sm">{type.label}</span>
              </label>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            {formData.test_type === 'one_shot'
              ? 'Single prompt/response test'
              : formData.test_type === 'multi_shot'
              ? 'Multiple sequential prompts to test context handling (needle in haystack, context window limits)'
              : 'Multi-turn conversation between doctor and patient'}
          </p>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">System Prompts (Optional)</label>
            <Link
              href="/prompts"
              className="text-xs text-primary hover:underline"
            >
              Manage Prompts
            </Link>
          </div>
          
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground">Doctor System Prompt</label>
            <select
              value={selectedDoctorSystemPromptId || ''}
              onChange={(e) =>
                setSelectedDoctorSystemPromptId(e.target.value ? parseInt(e.target.value) : undefined)
              }
              className="w-full rounded-lg border px-3 py-2 text-sm"
            >
              <option value="">Default doctor system prompt</option>
              {doctorSystemPrompts.map((prompt) => (
                <option key={prompt.id} value={prompt.id}>
                  {prompt.name}
                </option>
              ))}
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground">Patient System Prompt</label>
            <select
              value={selectedPatientSystemPromptId || ''}
              onChange={(e) =>
                setSelectedPatientSystemPromptId(e.target.value ? parseInt(e.target.value) : undefined)
              }
              className="w-full rounded-lg border px-3 py-2 text-sm"
            >
              <option value="">No system prompt (default behavior)</option>
              {patientSystemPrompts.map((prompt) => (
                <option key={prompt.id} value={prompt.id}>
                  {prompt.name}
                </option>
              ))}
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground">Test Prompt (Optional)</label>
            <select
              value={selectedPromptId || ''}
              onChange={(e) =>
                setSelectedPromptId(e.target.value ? parseInt(e.target.value) : undefined)
              }
              className="w-full rounded-lg border px-3 py-2 text-sm"
            >
              <option value="">None (use default prompts)</option>
              {prompts
                .filter((p) => p.prompt_type === 'test_prompt' && (!p.category || p.category === formData.test_type || ['multi_shot', 'conversation'].includes(formData.test_type)))
                .map((prompt) => (
                  <option key={prompt.id} value={prompt.id}>
                    {prompt.name} {prompt.category && `(${prompt.category})`}
                  </option>
                ))}
            </select>
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
            {formData.test_type === 'multi_shot' && (
              <div>
                <label className="text-sm font-medium">Number of Messages</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={formData.test_config?.num_messages || 10}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      test_config: {
                        ...formData.test_config,
                        num_messages: parseInt(e.target.value),
                      },
                    })
                  }
                  className="mt-1 w-full rounded border px-3 py-2 text-sm"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Number of sequential messages to send (for context window testing)
                </p>
                <div className="mt-2">
                  <label className="text-sm font-medium">Custom Prompts (Optional)</label>
                  <textarea
                    placeholder="Enter prompts, one per line. If empty, auto-generated messages will be used."
                    value={Array.isArray(formData.test_config?.prompts) 
                      ? formData.test_config.prompts.join('\n')
                      : ''}
                    onChange={(e) => {
                      const lines = e.target.value.split('\n').filter(l => l.trim())
                      setFormData({
                        ...formData,
                        test_config: {
                          ...formData.test_config,
                          prompts: lines.length > 0 ? lines : undefined,
                        },
                      })
                    }}
                    className="mt-1 w-full rounded border px-3 py-2 text-sm"
                    rows={4}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Custom prompts for needle-in-haystack or specific context tests
                  </p>
                </div>
              </div>
            )}
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
                <p className="text-xs text-muted-foreground mt-1">
                  Number of conversation turns between doctor and patient
                </p>
              </div>
            )}
            {formData.test_type === 'one_shot' && (
              <div>
                <label className="text-sm font-medium">Prompts (Optional)</label>
                <textarea
                  placeholder="Enter one or more prompts, one per line"
                  value={Array.isArray(formData.test_config?.prompts) 
                    ? formData.test_config.prompts.join('\n')
                    : formData.test_config?.prompt || ''}
                  onChange={(e) => {
                    const lines = e.target.value.split('\n').filter(l => l.trim())
                    setFormData({
                      ...formData,
                      test_config: {
                        ...formData.test_config,
                        prompts: lines.length > 1 ? lines : undefined,
                        prompt: lines.length === 1 ? lines[0] : undefined,
                      },
                    })
                  }}
                  className="mt-1 w-full rounded border px-3 py-2 text-sm"
                  rows={4}
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Enter prompts to test. Each line will be tested separately.
                </p>
              </div>
            )}
            <div className="space-y-2">
              <label className="text-sm font-medium">Chain-of-Thought (CoT) Reasoning</label>
              <div className="space-y-2">
                {formData.test_type === 'conversation' && (
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={formData.test_config?.enable_doctor_cot || false}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          test_config: {
                            ...formData.test_config,
                            enable_doctor_cot: e.target.checked,
                          },
                        })
                      }
                      className="rounded"
                    />
                    <span className="text-sm">Enable CoT for Doctor Model</span>
                  </label>
                )}
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={formData.test_config?.enable_patient_cot || false}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        test_config: {
                          ...formData.test_config,
                          enable_patient_cot: e.target.checked,
                        },
                      })
                    }
                    className="rounded"
                  />
                  <span className="text-sm">Enable CoT for Patient Model</span>
                </label>
                <p className="text-xs text-muted-foreground">
                  When enabled, the model will use ReACT-style reasoning (think-act-observe) before responding.
                </p>
              </div>
            </div>
            {false && formData.test_type === 'adversarial' && (
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
