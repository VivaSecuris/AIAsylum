import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { useCreateSuite, useBenchmarks, usePrompts, usePromptVariables } from '@/lib/hooks'
import { toast } from '@/lib/toast'
import { Plus, X, AlertTriangle } from 'lucide-react'
import Link from 'next/link'

const PROVIDERS = ['openai', 'anthropic', 'google', 'ollama']

const MODELS_BY_PROVIDER: Record<string, string[]> = {
  openai: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'],
  anthropic: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-2', 'claude-instant'],
  google: ['gemini-pro', 'gemini-pro-vision', 'palm-2'],
  ollama: ['llama2', 'llama3', 'llama3.2', 'mistral', 'mixtral', 'codellama'],
}

const TEST_TYPES = [
  { value: 'one_shot', label: 'One-Shot', desc: 'Single prompt/response test' },
  { value: 'multi_shot', label: 'Multi-Shot', desc: 'Multiple sequential prompts' },
  { value: 'conversation', label: 'Conversation', desc: 'Multi-turn conversation' },
  { value: 'group_therapy', label: 'Group Therapy', desc: 'Group therapy session with multiple patient models' },
  { value: 'benchmark', label: 'Benchmark', desc: 'Run standardized benchmark tests' },
]

interface Model {
  id: string
  provider: string
  model: string
}

export function SuiteForm() {
  const router = useRouter()
  const createSuite = useCreateSuite()
  const { data: benchmarksData } = useBenchmarks()
  const { data: prompts = [] } = usePrompts()
  const { data: doctorSystemPrompts = [] } = usePrompts({ prompt_type: 'system_prompt', target: 'doctor' })
  const { data: patientSystemPrompts = [] } = usePrompts({ prompt_type: 'system_prompt', target: 'patient' })

  const [suiteName, setSuiteName] = useState('')
  const [selectedTestType, setSelectedTestType] = useState<string>('')
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([])
  const [models, setModels] = useState<Model[]>([
    { id: 'model_1', provider: '', model: '' }
  ])
  const [numSamples, setNumSamples] = useState<number>(100)
  
  // Prompt selections
  const [selectedPromptId, setSelectedPromptId] = useState<number | undefined>(undefined)
  const [selectedPromptIds, setSelectedPromptIds] = useState<number[]>([]) // For multi-shot
  const [selectedDoctorSystemPromptId, setSelectedDoctorSystemPromptId] = useState<number | undefined>(undefined)
  const [selectedPatientSystemPromptId, setSelectedPatientSystemPromptId] = useState<number | undefined>(undefined)
  const [customPrompt, setCustomPrompt] = useState<string>('')
  const [customPrompts, setCustomPrompts] = useState<string[]>([''])
  const [variableValues, setVariableValues] = useState<Record<string, string>>({})
  
  // Get variables from selected prompt
  const { data: promptVariablesData } = usePromptVariables(selectedPromptId)
  const promptVariables = promptVariablesData?.variables || []
  
  // Reset prompt selections when test type changes
  useEffect(() => {
    if (selectedTestType !== 'one_shot') {
      setSelectedPromptId(undefined)
      setCustomPrompt('')
    }
    if (selectedTestType !== 'multi_shot') {
      setSelectedPromptIds([])
      setCustomPrompts([''])
    }
    if (selectedTestType !== 'conversation') {
      setSelectedDoctorSystemPromptId(undefined)
      setSelectedPatientSystemPromptId(undefined)
    }
    setVariableValues({})
  }, [selectedTestType])
  
  // Reset variable values when prompt changes
  useEffect(() => {
    if (promptVariables.length > 0) {
      const newValues: Record<string, string> = {}
      promptVariables.forEach((varName) => {
        newValues[varName] = variableValues[varName] || ''
      })
      setVariableValues(newValues)
    } else {
      setVariableValues({})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPromptId, promptVariablesData])
  
  // Check which test types need prompts
  const needsTestPrompts = selectedTestType === 'one_shot' || selectedTestType === 'multi_shot'
  const needsSystemPrompts = selectedTestType === 'conversation' || selectedTestType === 'group_therapy'
  const isBenchmarkTest = selectedTestType === 'benchmark'
  const isGroupTherapyTest = selectedTestType === 'group_therapy'

  // Calculate estimated test runs
  // For benchmark: selectedBenchmarks.length * models.length
  // For other test types: 1 * models.length (unless group_therapy which needs special handling)
  const estimatedRuns = isBenchmarkTest
    ? selectedBenchmarks.length * models.filter(m => m.provider && m.model).length
    : selectedTestType
    ? models.filter(m => m.provider && m.model).length
    : 0

  const addModel = () => {
    const newModel: Model = {
      id: `model_${Date.now()}`,
      provider: '',
      model: '',
    }
    setModels([...models, newModel])
  }

  const removeModel = (id: string) => {
    setModels(models.filter((m) => m.id !== id))
  }

  const updateModel = (id: string, field: 'provider' | 'model', value: string) => {
    setModels(
      models.map((m) => {
        if (m.id === id) {
          const updated = { ...m, [field]: value }
          if (field === 'provider' && m.model && !MODELS_BY_PROVIDER[value]?.includes(m.model)) {
            updated.model = ''
          }
          return updated
        }
        return m
      })
    )
  }


  const toggleBenchmark = (benchmark: string) => {
    setSelectedBenchmarks(prev =>
      prev.includes(benchmark)
        ? prev.filter(b => b !== benchmark)
        : [...prev, benchmark]
    )
  }

  const getAvailableModels = (provider: string): string[] => {
    return MODELS_BY_PROVIDER[provider] || []
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // Validation
    if (!selectedTestType) {
      toast.error('Please select a test type')
      return
    }

    if (isBenchmarkTest && selectedBenchmarks.length === 0) {
      toast.error('Please select at least one benchmark')
      return
    }

    const validModels = models.filter(m => m.provider && m.model)
    if (validModels.length === 0) {
      toast.error('Please add at least one model')
      return
    }

    if (estimatedRuns > 50) {
      const confirmed = window.confirm(
        `This will create ${estimatedRuns} test runs. This may take a long time. Continue?`
      )
      if (!confirmed) return
    }

    try {
      // Build test_config with prompt selections
      const testConfig: Record<string, any> = {}
      
      // Add prompts for one-shot/multi-shot tests
      if (needsTestPrompts) {
        if (selectedTestType === 'one_shot') {
          if (selectedPromptId) {
            testConfig.prompt_id = selectedPromptId
          } else if (customPrompt) {
            testConfig.prompt = customPrompt
          }
        } else if (selectedTestType === 'multi_shot') {
          if (selectedPromptIds.length > 0) {
            testConfig.prompt_id = selectedPromptIds[0]
          } else if (customPrompts.length > 0 && customPrompts[0]) {
            testConfig.prompts = customPrompts.filter(p => p.trim())
          }
        }
      }
      
      // Add system prompts for conversation tests
      if (needsSystemPrompts) {
        if (selectedDoctorSystemPromptId) {
          testConfig.doctor_system_prompt_id = selectedDoctorSystemPromptId
        }
        if (selectedPatientSystemPromptId) {
          testConfig.patient_system_prompt_id = selectedPatientSystemPromptId
        }
      }
      
      // Add variables if any are set
      if (Object.keys(variableValues).length > 0 && Object.values(variableValues).some(v => v.trim())) {
        testConfig.variables = variableValues
      }

      // For benchmark test type, we need to pass benchmarks
      // For other test types, benchmarks array should be empty
      const benchmarksToUse = isBenchmarkTest ? selectedBenchmarks : []

      const result = await createSuite.mutateAsync({
        name: suiteName || undefined,
        test_types: [selectedTestType],
        benchmarks: benchmarksToUse,
        models: validModels.map(m => ({ provider: m.provider, model: m.model })),
        test_config: Object.keys(testConfig).length > 0 ? testConfig : undefined,
        num_samples: isBenchmarkTest ? numSamples : undefined,
      })

      toast.success(`Suite created! Starting ${estimatedRuns} test runs...`)
      router.push(`/suite/${result.id}`)
    } catch (error) {
      console.error('Failed to create suite:', error)
      toast.error('Failed to create suite. Please check your configuration.')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="rounded-lg border bg-card p-6 space-y-6">
        <h2 className="text-xl font-semibold">Suite Configuration</h2>

        {/* Suite Name */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Suite Name (Optional)</label>
          <input
            type="text"
            value={suiteName}
            onChange={(e) => setSuiteName(e.target.value)}
            placeholder="My Test Suite"
            className="w-full rounded-lg border px-3 py-2 text-sm"
          />
        </div>

        {/* Test Type Selection (Radio) */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Test Type</label>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-3">
            {TEST_TYPES.map((type) => (
              <label
                key={type.value}
                className={`flex items-start gap-2 rounded-lg border p-3 cursor-pointer transition-colors ${
                  selectedTestType === type.value
                    ? 'bg-primary/10 border-primary'
                    : 'bg-muted/30 hover:bg-muted/50'
                }`}
              >
                <input
                  type="radio"
                  name="test_type"
                  value={type.value}
                  checked={selectedTestType === type.value}
                  onChange={(e) => setSelectedTestType(e.target.value)}
                  className="mt-1"
                />
                <div>
                  <div className="font-medium text-sm">{type.label}</div>
                  <div className="text-xs text-muted-foreground">{type.desc}</div>
                </div>
              </label>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Select one test type to run against all selected models
          </p>
        </div>

        {/* Benchmarks Selection (only show if benchmark test type is selected) */}
        {isBenchmarkTest && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Benchmarks</label>
            {benchmarksData?.benchmarks && benchmarksData.benchmarks.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {benchmarksData.benchmarks.map((benchmark: any) => (
                  <label
                    key={benchmark.name}
                    className={`flex items-start gap-2 rounded-lg border p-3 cursor-pointer transition-colors ${
                      selectedBenchmarks.includes(benchmark.name)
                        ? 'bg-primary/10 border-primary'
                        : 'bg-muted/30 hover:bg-muted/50'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedBenchmarks.includes(benchmark.name)}
                      onChange={() => toggleBenchmark(benchmark.name)}
                      className="mt-1 rounded"
                    />
                    <div>
                      <div className="font-medium text-sm capitalize">{benchmark.name}</div>
                      <div className="text-xs text-muted-foreground">{benchmark.description}</div>
                    </div>
                  </label>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Loading benchmarks...</p>
            )}
          </div>
        )}

        {/* Models Selection */}
        <div className="space-y-3">
          <label className="text-sm font-medium">Models</label>
          {models.map((model, index) => (
            <div key={model.id} className="rounded-lg border bg-card p-4">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium">Model {index + 1}</label>
                {models.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeModel(model.id)}
                    className="text-red-600 hover:text-red-800"
                    title="Remove model"
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-muted-foreground">Provider</label>
                  <select
                    value={model.provider}
                    onChange={(e) => updateModel(model.id, 'provider', e.target.value)}
                    className="mt-1 w-full rounded border px-3 py-2 text-sm"
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
                    value={model.model}
                    onChange={(e) => updateModel(model.id, 'model', e.target.value)}
                    disabled={!model.provider}
                    className="mt-1 w-full rounded border px-3 py-2 text-sm disabled:bg-muted"
                  >
                    <option value="">Select model</option>
                    {getAvailableModels(model.provider).map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          ))}
          <button
            type="button"
            onClick={addModel}
            className="flex items-center gap-2 rounded-lg border border-dashed px-4 py-2 text-sm font-medium hover:bg-muted"
          >
            <Plus className="h-4 w-4" />
            Add Model
          </button>
        </div>

        {/* Number of Samples (for benchmarks) */}
        {isBenchmarkTest && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Number of Samples</label>
            <input
              type="number"
              min="1"
              max="10000"
              value={numSamples}
              onChange={(e) => setNumSamples(parseInt(e.target.value) || 100)}
              className="w-full rounded-lg border px-3 py-2 text-sm"
            />
            <p className="text-xs text-muted-foreground">
              Number of test samples to run from each benchmark dataset
            </p>
          </div>
        )}

        {/* Group Therapy Patients (if group therapy is selected) */}
        {isGroupTherapyTest && (
          <div className="space-y-3 rounded-lg border bg-muted/30 p-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Patient Models for Group Therapy</label>
              <p className="text-xs text-muted-foreground">
                The models selected above will be used as the doctor. Add patient models below.
              </p>
            </div>
            <p className="text-xs text-muted-foreground mb-2">
              Note: For group therapy, you need to select a doctor model above, and add patient models below.
              The suite will create one test run per patient model combination.
            </p>
            <p className="text-xs text-yellow-600">
              ⚠️ Group therapy in suites: Each model above will be used as the doctor, and each patient model below will be tested separately.
            </p>
          </div>
        )}

        {/* Prompt Selection for Test Types */}
        {needsTestPrompts && (
          <div className="space-y-4 rounded-lg border bg-muted/30 p-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                {selectedTestType === 'one_shot'
                  ? 'Test Prompt (One-Shot)'
                  : 'Test Prompts (Multi-Shot)'}
              </label>
              <Link
                href="/prompts"
                className="text-xs text-primary hover:underline"
              >
                Manage Prompts
              </Link>
            </div>

            {selectedTestType === 'one_shot' ? (
              // One-shot: single prompt
              <div className="space-y-3">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-muted-foreground">Select Prompt from Library (Optional)</label>
                  <select
                    value={selectedPromptId || ''}
                    onChange={(e) => {
                      const newPromptId = e.target.value ? parseInt(e.target.value) : undefined
                      setSelectedPromptId(newPromptId)
                      if (newPromptId) {
                        setCustomPrompt('')
                      }
                    }}
                    className="w-full rounded-lg border px-3 py-2 text-sm"
                  >
                    <option value="">None (use custom prompt below)</option>
                    {prompts
                      .filter((p) => p.prompt_type === 'test_prompt')
                      .map((prompt) => (
                        <option key={prompt.id} value={prompt.id}>
                          {prompt.name} {prompt.category && `(${prompt.category})`}
                        </option>
                      ))}
                  </select>
                </div>

                {!selectedPromptId && (
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-muted-foreground">Or Enter Custom Prompt</label>
                    <textarea
                      placeholder="Enter a single prompt to test"
                      value={customPrompt}
                      onChange={(e) => {
                        setCustomPrompt(e.target.value)
                        if (e.target.value) {
                          setSelectedPromptId(undefined)
                        }
                      }}
                      className="w-full rounded-lg border px-3 py-2 text-sm"
                      rows={4}
                    />
                  </div>
                )}

                {promptVariables.length > 0 && (
                  <div className="mt-3 space-y-2 rounded-lg border bg-background p-3">
                    <label className="text-xs font-medium">Prompt Variables</label>
                    {promptVariables.map((varName) => (
                      <div key={varName} className="space-y-1">
                        <label className="text-xs font-medium text-muted-foreground">
                          ${varName}
                        </label>
                        <input
                          type="text"
                          value={variableValues[varName] || ''}
                          onChange={(e) =>
                            setVariableValues({
                              ...variableValues,
                              [varName]: e.target.value,
                            })
                          }
                          className="w-full rounded border px-2 py-1 text-sm"
                          placeholder={`Enter value for ${varName}`}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : selectedTestType === 'multi_shot' ? (
              // Multi-shot: multiple prompts
              <div className="space-y-3">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-muted-foreground">Select Prompt from Library (Optional)</label>
                  <select
                    value={selectedPromptIds[0] || ''}
                    onChange={(e) => {
                      const newPromptId = e.target.value ? parseInt(e.target.value) : undefined
                      if (newPromptId) {
                        setSelectedPromptIds([newPromptId])
                        setCustomPrompts([''])
                      } else {
                        setSelectedPromptIds([])
                      }
                    }}
                    className="w-full rounded-lg border px-3 py-2 text-sm"
                  >
                    <option value="">None (use custom prompts below)</option>
                    {prompts
                      .filter((p) => p.prompt_type === 'test_prompt')
                      .map((prompt) => (
                        <option key={prompt.id} value={prompt.id}>
                          {prompt.name} {prompt.category && `(${prompt.category})`}
                        </option>
                      ))}
                  </select>
                </div>

                {selectedPromptIds.length === 0 && (
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-muted-foreground">Or Enter Custom Prompts (one per line)</label>
                    <textarea
                      placeholder="Enter multiple prompts, one per line"
                      value={customPrompts.join('\n')}
                      onChange={(e) => {
                        const lines = e.target.value.split('\n').filter(l => l.trim())
                        setCustomPrompts(lines.length > 0 ? lines : [''])
                        if (e.target.value) {
                          setSelectedPromptIds([])
                        }
                      }}
                      className="w-full rounded-lg border px-3 py-2 text-sm"
                      rows={6}
                    />
                    <p className="text-xs text-muted-foreground">
                      Enter multiple prompts, one per line. Each prompt will be sent sequentially.
                    </p>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        )}

        {/* System Prompts for Conversation Tests */}
        {needsSystemPrompts && (
          <div className="space-y-4 rounded-lg border bg-muted/30 p-4">
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
          </div>
        )}

        {/* Estimated Runs */}
        <div className="rounded-lg border bg-muted/30 p-4">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-sm font-semibold">Estimated Test Runs</h3>
            {estimatedRuns > 50 && (
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
            )}
          </div>
          <p className="text-2xl font-bold">{estimatedRuns}</p>
          <p className="text-xs text-muted-foreground mt-1">
            {isBenchmarkTest ? (
              <>
                {selectedBenchmarks.length} benchmark{selectedBenchmarks.length !== 1 ? 's' : ''} × {models.filter(m => m.provider && m.model).length} model{models.filter(m => m.provider && m.model).length !== 1 ? 's' : ''}
              </>
            ) : selectedTestType ? (
              <>
                1 test type ({selectedTestType}) × {models.filter(m => m.provider && m.model).length} model{models.filter(m => m.provider && m.model).length !== 1 ? 's' : ''}
              </>
            ) : (
              'Select a test type to see estimated runs'
            )}
          </p>
          {estimatedRuns > 50 && (
            <p className="text-xs text-yellow-600 mt-2">
              Warning: This will create a large number of test runs and may take a long time to complete.
            </p>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={createSuite.isPending || estimatedRuns === 0}
          className="w-full rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {createSuite.isPending ? 'Creating Suite...' : 'Create Suite'}
        </button>
      </div>
    </form>
  )
}
