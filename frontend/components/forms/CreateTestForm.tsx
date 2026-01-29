import { useState, useEffect } from 'react'
import { ModelSelector } from './ModelSelector'
import { MultiPatientSelector } from './MultiPatientSelector'
import { TestRunRequest } from '@/lib/api'
import { useCreateTestRun, usePrompts, usePromptVariables, useBenchmarks, useRunBenchmark } from '@/lib/hooks'
import { useRouter } from 'next/router'
import { toast } from '@/lib/toast'
import Link from 'next/link'
import { AnalysisConfig } from '@/components/analysis/AnalysisConfigDialog'

export function CreateTestForm() {
  const router = useRouter()
  const createTestRun = useCreateTestRun()
  const runBenchmark = useRunBenchmark()
  const { data: prompts = [] } = usePrompts()
  const { data: benchmarksData } = useBenchmarks()

  // Check if benchmark type is requested via query param
  const initialTestType = router.query.type === 'benchmark' ? 'benchmark' : 'one_shot'
  
  // Check if prompt ID is provided via query param
  const promptIdFromQuery = router.query.promptId 
    ? parseInt(Array.isArray(router.query.promptId) ? router.query.promptId[0] : router.query.promptId)
    : undefined

  const [formData, setFormData] = useState<TestRunRequest>({
    doctor_provider: '',
    doctor_model: '',
    patient_provider: '',
    patient_model: '',
    test_type: initialTestType,
    test_config: {},
  })
  
  // For one-shot and multi-shot, we only need one model (patient)
  // For conversation, we need both doctor and patient
  // For group_therapy, we need doctor and multiple patients
  // For benchmark, we need one model and benchmark selection
  const isConversationTest = formData.test_type === 'conversation'
  const isGroupTherapyTest = formData.test_type === 'group_therapy'
  const isOneShotOrMultiShot = formData.test_type === 'one_shot' || formData.test_type === 'multi_shot'
  const isBenchmarkTest = formData.test_type === 'benchmark'

  // Initialize prompt selection with query parameter if provided
  const [selectedPromptId, setSelectedPromptId] = useState<number | undefined>(promptIdFromQuery)
  const [selectedPromptIds, setSelectedPromptIds] = useState<number[]>(
    promptIdFromQuery ? [promptIdFromQuery] : []
  ) // For multi-shot: multiple prompts
  const [selectedDoctorSystemPromptId, setSelectedDoctorSystemPromptId] = useState<number | undefined>(undefined)
  const [selectedPatientSystemPromptId, setSelectedPatientSystemPromptId] = useState<number | undefined>(undefined)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [variableValues, setVariableValues] = useState<Record<string, string>>({})
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('')
  const [numSamples, setNumSamples] = useState<number>(100)
  const [groupTherapyPatients, setGroupTherapyPatients] = useState<Array<{ id: string; provider: string; model: string }>>([
    { id: 'patient_1', provider: '', model: '' }
  ])
  
  // Auto-analysis configuration
  const [enableAutoAnalysis, setEnableAutoAnalysis] = useState(false)
  const [showAnalysisConfig, setShowAnalysisConfig] = useState(false)
  const [analysisConfig, setAnalysisConfig] = useState<AnalysisConfig>({
    enable_cot_detection: true,
    cot_analysis_mode: 'full',
    enable_factuality_check: false,
    enable_manipulation_analysis: false,
  })
  
  // Get system prompts separately
  const { data: doctorSystemPrompts = [] } = usePrompts({ prompt_type: 'system_prompt', target: 'doctor' })
  const { data: patientSystemPrompts = [] } = usePrompts({ prompt_type: 'system_prompt', target: 'patient' })
  
  // Get variables from selected prompt
  const { data: promptVariablesData } = usePromptVariables(selectedPromptId)
  const promptVariables = promptVariablesData?.variables || []
  
  // Update test type if query param changes
  useEffect(() => {
    if (router.query.type === 'benchmark' && formData.test_type !== 'benchmark') {
      setFormData({ ...formData, test_type: 'benchmark' })
    }
  }, [router.query.type])
  
  // Pre-select prompt from query parameter when prompts are loaded or query changes
  useEffect(() => {
    // Re-read promptId from query in case it changed
    const currentPromptId = router.query.promptId 
      ? parseInt(Array.isArray(router.query.promptId) ? router.query.promptId[0] : router.query.promptId)
      : undefined
    
    if (currentPromptId && prompts.length > 0) {
      const prompt = prompts.find(p => p.id === currentPromptId)
      if (prompt && prompt.prompt_type === 'test_prompt') {
        // Set the prompt selection based on test type
        if (formData.test_type === 'one_shot') {
          setSelectedPromptId(currentPromptId)
          setSelectedPromptIds([]) // Clear multi-shot selection
        } else if (formData.test_type === 'multi_shot') {
          setSelectedPromptIds([currentPromptId])
          setSelectedPromptId(undefined) // Clear one-shot selection
        } else {
          // Default to one_shot for single prompt
          setFormData(prev => ({ ...prev, test_type: 'one_shot' }))
          setSelectedPromptId(currentPromptId)
          setSelectedPromptIds([])
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router.query.promptId, prompts.length])
  
  // Reset variable values when prompt changes
  useEffect(() => {
    if (promptVariables.length > 0) {
      // Initialize empty values for new variables, keep existing values for variables that still exist
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
  
  // Reset prompt selections when test type changes (but preserve if coming from query param)
  useEffect(() => {
    // Don't reset if we have a promptId from query and prompts are loaded
    if (promptIdFromQuery && prompts.length > 0) {
      const prompt = prompts.find(p => p.id === promptIdFromQuery)
      if (prompt && prompt.prompt_type === 'test_prompt') {
        // Keep the prompt selected, just adjust based on test type
        if (formData.test_type === 'one_shot') {
          setSelectedPromptId(promptIdFromQuery)
          setSelectedPromptIds([])
        } else if (formData.test_type === 'multi_shot') {
          setSelectedPromptIds([promptIdFromQuery])
          setSelectedPromptId(undefined)
        }
        return // Don't reset if we have a valid prompt from query
      }
    }
    
    // Otherwise, reset as normal
    if (formData.test_type === 'one_shot') {
      setSelectedPromptIds([])
    } else if (formData.test_type === 'multi_shot') {
      setSelectedPromptId(undefined)
    } else if (formData.test_type === 'conversation') {
      setSelectedPromptId(undefined)
      setSelectedPromptIds([])
    } else if (formData.test_type === 'group_therapy') {
      setSelectedPromptId(undefined)
      setSelectedPromptIds([])
      // Initialize with one patient if empty
      if (groupTherapyPatients.length === 0) {
        setGroupTherapyPatients([{ id: 'patient_1', provider: '', model: '' }])
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [formData.test_type])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      // Handle benchmark submission differently
      if (isBenchmarkTest) {
        if (!selectedBenchmark || !formData.patient_provider || !formData.patient_model) {
          const missing = []
          if (!selectedBenchmark) missing.push('benchmark')
          if (!formData.patient_provider) missing.push('provider')
          if (!formData.patient_model) missing.push('model')
          toast.error(`Please select: ${missing.join(', ')}`)
          return
        }

        const payload = {
          provider: formData.patient_provider,
          model: formData.patient_model,
          benchmark: selectedBenchmark,
          num_samples: numSamples,
        }
        
        const result = await runBenchmark.mutateAsync(payload)
        const testRunId = result.test_run_id || result.id
        toast.success(`Benchmark started! Test run ID: ${testRunId}`)
        router.push(`/test-runs/${testRunId}`)
        return
      }

      // Handle regular test run submission
      const testConfig = {
        ...formData.test_config,
      }
      
      // Handle group therapy: add patients array
      if (isGroupTherapyTest) {
        // Validate at least one patient is selected
        const validPatients = groupTherapyPatients.filter(
          (p) => p.provider && p.model
        )
        if (validPatients.length === 0) {
          toast.error('Please add at least one patient model for the group therapy session')
          return
        }
        // Convert to format expected by backend
        testConfig.patients = validPatients.map((p) => ({
          provider: p.provider,
          model: p.model,
        }))
      }
      
      // Handle prompt selection based on test type
      if (formData.test_type === 'multi_shot') {
        // For multi-shot: use custom prompts if provided, otherwise use first selected library prompt
        if (testConfig.prompts && testConfig.prompts.length > 0) {
          // Custom prompts take precedence
          // prompt_id will be ignored if prompts array is provided
        } else if (selectedPromptIds.length > 0) {
          // Use first selected prompt as prompt_id
          // Note: For multiple library prompts, user should copy text to custom prompts field
          testConfig.prompt_id = selectedPromptIds[0]
        }
      } else if (selectedPromptId) {
        // For one-shot: use selected prompt
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
        test_config: {
          ...testConfig,
          // Store auto-analysis config if enabled
          ...(enableAutoAnalysis ? {
            auto_analysis: true,
            analysis_config: analysisConfig,
          } : {}),
        },
        variables: Object.keys(variableValues).length > 0 ? variableValues : undefined,
      }
      const result = await createTestRun.mutateAsync(submitData)
      
      // If auto-analysis is enabled, inform the user
      if (enableAutoAnalysis) {
        toast.success(`Test run created! Analysis will start automatically when the test completes.`)
      }
      
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

        {/* Model Selectors - conditional based on test type */}
        {isConversationTest || isGroupTherapyTest ? (
          <>
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

            {isGroupTherapyTest ? (
              <MultiPatientSelector
                patients={groupTherapyPatients}
                onChange={setGroupTherapyPatients}
              />
            ) : (
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
            )}
          </>
        ) : (
          // For one-shot and multi-shot, only need one model (used as patient)
          <ModelSelector
            label="Model"
            provider={formData.patient_provider}
            model={formData.patient_model}
            onProviderChange={(provider) =>
              setFormData({ 
                ...formData, 
                patient_provider: provider,
                // Also set doctor to same for consistency (though not used)
                doctor_provider: provider,
              })
            }
            onModelChange={(model) =>
              setFormData({ 
                ...formData, 
                patient_model: model,
                // Also set doctor to same for consistency (though not used)
                doctor_model: model,
              })
            }
          />
        )}

        <div className="space-y-2">
          <label className="text-sm font-medium">Test Type</label>
          <div className="flex gap-4">
            {[
              { value: 'one_shot', label: 'One-Shot', desc: 'Single prompt/response test' },
              { value: 'multi_shot', label: 'Multi-Shot', desc: 'Multiple sequential prompts to test context handling' },
              { value: 'conversation', label: 'Conversation', desc: 'Multi-turn conversation between doctor and patient' },
              { value: 'group_therapy', label: 'Group Therapy', desc: 'Group therapy session with multiple patient models sharing in a led discussion' },
              { value: 'benchmark', label: 'Benchmark', desc: 'Run standardized benchmark tests' },
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
              : formData.test_type === 'conversation'
              ? 'Multi-turn conversation between doctor and patient'
              : formData.test_type === 'group_therapy'
              ? 'Group therapy session with multiple patient models sharing in a led discussion'
              : 'Run standardized benchmark tests on the model'}
          </p>
        </div>

        {/* Prompts section - different for each test type */}
        {isBenchmarkTest ? (
          // Benchmark test: show benchmark selection
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Benchmark</label>
              <select
                value={selectedBenchmark}
                onChange={(e) => setSelectedBenchmark(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="" className="bg-background text-foreground">
                  Select benchmark
                </option>
                {benchmarksData?.benchmarks?.map((benchmark: any) => (
                  <option
                    key={benchmark.name}
                    value={benchmark.name}
                    className="bg-background text-foreground"
                  >
                    {benchmark.name} - {benchmark.description}
                  </option>
                ))}
              </select>
              {selectedBenchmark && (
                <p className="text-xs text-muted-foreground">
                  {benchmarksData?.benchmarks?.find((b: any) => b.name === selectedBenchmark)?.description}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Number of Samples</label>
              <input
                type="number"
                min="1"
                max="10000"
                value={numSamples}
                onChange={(e) => setNumSamples(parseInt(e.target.value) || 100)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              />
              <p className="text-xs text-muted-foreground">
                Number of test samples to run from the benchmark dataset
              </p>
            </div>

            {benchmarksData?.benchmarks && benchmarksData.benchmarks.length > 0 && (
              <div className="mt-4 rounded-lg border bg-muted/30 p-4">
                <h3 className="text-sm font-semibold mb-2">Available Benchmarks</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {benchmarksData.benchmarks.map((benchmark: any) => (
                    <div
                      key={benchmark.name}
                      className={`rounded border p-3 text-sm cursor-pointer transition-colors ${
                        selectedBenchmark === benchmark.name
                          ? 'bg-primary/10 border-primary'
                          : 'bg-muted/30 hover:bg-muted/50'
                      }`}
                      onClick={() => setSelectedBenchmark(benchmark.name)}
                    >
                      <h4 className="font-medium capitalize">{benchmark.name}</h4>
                      <p className="text-xs text-muted-foreground mt-1">{benchmark.description}</p>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Click on a benchmark above to select it, or choose from the dropdown above.
                </p>
              </div>
            )}
          </div>
        ) : isConversationTest || isGroupTherapyTest ? (
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
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
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
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
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
        ) : (
          // For one-shot and multi-shot: prompt selection/input
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                {formData.test_type === 'one_shot' ? 'Test Prompt' : 'Test Prompts'}
              </label>
              <Link
                href="/prompts"
                className="text-xs text-primary hover:underline"
              >
                Manage Prompts
              </Link>
            </div>
            
            {formData.test_type === 'one_shot' ? (
              // One-shot: single prompt (from library or custom)
              <div className="space-y-3">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-muted-foreground">Select Prompt from Library (Optional)</label>
                  <select
                    value={selectedPromptId || ''}
                    onChange={(e) => {
                      const newPromptId = e.target.value ? parseInt(e.target.value) : undefined
                      setSelectedPromptId(newPromptId)
                      // Clear custom prompt when selecting from library
                      if (newPromptId) {
                        setFormData({
                          ...formData,
                          test_config: {
                            ...formData.test_config,
                            prompt: undefined,
                            prompts: undefined,
                          },
                        })
                        // Update URL to reflect selection (but don't push to avoid navigation)
                        router.replace({
                          pathname: router.pathname,
                          query: { ...router.query, promptId: newPromptId },
                        }, undefined, { shallow: true })
                      } else {
                        // Remove promptId from URL if deselected
                        const newQuery = { ...router.query }
                        delete newQuery.promptId
                        router.replace({
                          pathname: router.pathname,
                          query: newQuery,
                        }, undefined, { shallow: true })
                      }
                    }}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
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
                  {promptIdFromQuery && selectedPromptId === promptIdFromQuery && (
                    <p className="text-xs text-primary mt-1">
                      ✓ Prompt pre-selected from prompt library
                    </p>
                  )}
                </div>
                
                {!selectedPromptId && (
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-muted-foreground">Or Enter Custom Prompt</label>
                    <textarea
                      placeholder="Enter a single prompt to test"
                      value={formData.test_config?.prompt || ''}
                      onChange={(e) => {
                        const value = e.target.value || undefined
                        setFormData({
                          ...formData,
                          test_config: {
                            ...formData.test_config,
                            prompt: value,
                            prompts: undefined, // Clear prompts array for one-shot
                          },
                        })
                        // Clear library selection when entering custom prompt
                        if (value) {
                          setSelectedPromptId(undefined)
                        }
                      }}
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      rows={4}
                    />
                    <p className="text-xs text-muted-foreground">
                      Enter a single prompt to test the model's response
                    </p>
                  </div>
                )}
                
                {promptVariables.length > 0 && (
                  <div className="mt-3 space-y-2 rounded-lg border bg-muted/30 p-3">
                    <label className="text-xs font-medium">Prompt Variables</label>
                    <p className="text-xs text-muted-foreground mb-2">
                      Fill in the values for variables used in this prompt (e.g., $country, $name)
                    </p>
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
                          placeholder={`Enter value for ${varName}`}
                          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              // Multi-shot: multiple prompts (from library or custom)
              <div className="space-y-3">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-muted-foreground">Select Prompts from Library (Optional)</label>
                  <select
                    multiple
                    size={5}
                    value={selectedPromptIds.map(String)}
                    onChange={(e) => {
                      const selectedIds = Array.from(e.target.selectedOptions, (opt) => parseInt(opt.value))
                      setSelectedPromptIds(selectedIds)
                      // Clear custom prompts when selecting from library
                      if (selectedIds.length > 0) {
                        setFormData({
                          ...formData,
                          test_config: {
                            ...formData.test_config,
                            prompts: undefined,
                          },
                        })
                      }
                    }}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    {prompts
                      .filter((p) => p.prompt_type === 'test_prompt')
                      .map((prompt) => (
                        <option key={prompt.id} value={prompt.id}>
                          {prompt.name} {prompt.category && `(${prompt.category})`}
                        </option>
                      ))}
                  </select>
                  <p className="text-xs text-muted-foreground">
                    Hold Ctrl/Cmd to select multiple prompts. Selected: {selectedPromptIds.length} prompt(s)
                  </p>
                  {selectedPromptIds.length > 0 && (
                    <div className="mt-2 p-2 bg-muted/50 rounded text-xs">
                      <strong>Selected prompts ({selectedPromptIds.length}):</strong>
                      <ul className="list-disc list-inside mt-1">
                        {selectedPromptIds.map((id) => {
                          const prompt = prompts.find((p) => p.id === id)
                          return <li key={id}>{prompt?.name || `ID: ${id}`}</li>
                        })}
                      </ul>
                      <p className="mt-2 text-muted-foreground">
                        Note: Only the first selected prompt will be used. To use multiple prompts, copy their text to the custom prompts field below.
                      </p>
                    </div>
                  )}
                </div>
                
                <div className="space-y-2">
                  <label className="text-xs font-medium text-muted-foreground">
                    Or Enter Custom Prompts (One per line)
                    {selectedPromptIds.length > 0 && ' - Custom prompts will override selected library prompts'}
                  </label>
                  <textarea
                    placeholder="Enter multiple prompts, one per line. Each line will be sent sequentially to test context handling."
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
                      // Clear library selection when entering custom prompts
                      if (lines.length > 0) {
                        setSelectedPromptIds([])
                      }
                    }}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                    rows={6}
                  />
                  <p className="text-xs text-muted-foreground">
                    Each line will be sent as a separate prompt in sequence. Useful for context window testing and needle-in-haystack scenarios.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Auto-Analysis Section */}
        <div className="space-y-4 rounded-lg border bg-muted/30 p-4">
          <label className="flex items-center gap-3">
            <input
              type="checkbox"
              checked={enableAutoAnalysis}
              onChange={(e) => {
                setEnableAutoAnalysis(e.target.checked)
                if (e.target.checked) {
                  setShowAnalysisConfig(true)
                }
              }}
              className="rounded"
            />
            <div className="flex-1">
              <span className="text-sm font-medium">Enable Automatic Analysis</span>
              <p className="text-xs text-muted-foreground">
                Automatically run analysis when the test completes. Configure analysis options below.
              </p>
            </div>
          </label>

          {enableAutoAnalysis && (
            <div className="mt-4 space-y-4 rounded-lg border bg-card p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold">Analysis Configuration</h3>
                <button
                  type="button"
                  onClick={() => setShowAnalysisConfig(!showAnalysisConfig)}
                  className="text-xs text-primary hover:underline"
                >
                  {showAnalysisConfig ? 'Hide' : 'Show'} Options
                </button>
              </div>

              {showAnalysisConfig && (
                <div className="space-y-4">
                  {/* Evaluator Model Selection */}
                  <div className="space-y-2">
                    <label className="text-xs font-medium text-muted-foreground">Evaluator Model (Optional)</label>
                    <p className="text-xs text-muted-foreground mb-2">
                      Select an AI model to analyze the conversation. If not specified, the doctor model will be used.
                    </p>
                    <ModelSelector
                      label=""
                      provider={analysisConfig.evaluator_provider || ''}
                      model={analysisConfig.evaluator_model || ''}
                      onProviderChange={(provider) =>
                        setAnalysisConfig({ ...analysisConfig, evaluator_provider: provider || undefined })
                      }
                      onModelChange={(model) =>
                        setAnalysisConfig({ ...analysisConfig, evaluator_model: model || undefined })
                      }
                    />
                  </div>

                  {/* Chain of Thought Detection */}
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={analysisConfig.enable_cot_detection}
                        onChange={(e) =>
                          setAnalysisConfig({ ...analysisConfig, enable_cot_detection: e.target.checked })
                        }
                        className="rounded"
                      />
                      <span className="text-sm font-medium">Enable Chain of Thought Detection</span>
                    </label>
                    {analysisConfig.enable_cot_detection && (
                      <div className="ml-6 space-y-2">
                        <label className="text-xs font-medium text-muted-foreground">Analysis Mode</label>
                        <select
                          value={analysisConfig.cot_analysis_mode}
                          onChange={(e) =>
                            setAnalysisConfig({
                              ...analysisConfig,
                              cot_analysis_mode: e.target.value as 'full' | 'partial' | 'none',
                            })
                          }
                          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                        >
                          <option value="full">Full Analysis</option>
                          <option value="partial">Partial Analysis</option>
                          <option value="none">None</option>
                        </select>
                      </div>
                    )}
                  </div>

                  {/* Factuality Check */}
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={analysisConfig.enable_factuality_check}
                      onChange={(e) =>
                        setAnalysisConfig({ ...analysisConfig, enable_factuality_check: e.target.checked })
                      }
                      className="rounded"
                    />
                    <span className="text-sm font-medium">Enable Factuality/Hallucination Detection</span>
                  </label>

                  {/* Manipulation Analysis */}
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={analysisConfig.enable_manipulation_analysis}
                      onChange={(e) =>
                        setAnalysisConfig({ ...analysisConfig, enable_manipulation_analysis: e.target.checked })
                      }
                      className="rounded"
                    />
                    <span className="text-sm font-medium">Enable Manipulation Analysis</span>
                  </label>
                </div>
              )}
            </div>
          )}
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
                <label className="text-sm font-medium">Number of Messages (if not using custom prompts)</label>
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
                  className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  disabled={Array.isArray(formData.test_config?.prompts) && formData.test_config.prompts.length > 0}
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Number of auto-generated sequential messages (only used if custom prompts are not provided)
                </p>
              </div>
            )}
            {(formData.test_type === 'conversation' || formData.test_type === 'group_therapy') && (
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
                  className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Number of conversation turns {formData.test_type === 'group_therapy' ? 'in the group therapy session' : 'between doctor and patient'}
                </p>
              </div>
            )}
            <div className="space-y-2">
              <label className="text-sm font-medium">Chain-of-Thought (CoT) Reasoning</label>
              <div className="space-y-2">
                {isConversationTest && (
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
                  <span className="text-sm">Enable CoT for {isConversationTest ? 'Patient' : 'Model'}</span>
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
                  className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
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
            (isBenchmarkTest
              ? runBenchmark.isPending || !selectedBenchmark || !formData.patient_provider || !formData.patient_model
              : createTestRun.isPending ||
                (isGroupTherapyTest
                  ? !formData.doctor_provider || !formData.doctor_model || groupTherapyPatients.filter((p) => p.provider && p.model).length === 0
                  : !formData.patient_provider ||
                    !formData.patient_model ||
                    (isConversationTest && (!formData.doctor_provider || !formData.doctor_model))))
          }
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {isBenchmarkTest
            ? runBenchmark.isPending
              ? 'Running Benchmark...'
              : 'Run Benchmark'
            : createTestRun.isPending
            ? 'Creating...'
            : 'Create Test Run'}
        </button>
      </div>
    </form>
  )
}
