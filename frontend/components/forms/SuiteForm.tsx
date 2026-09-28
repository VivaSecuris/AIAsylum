import { useMemo, useState } from 'react'
import { useRouter } from 'next/router'
import { AlertTriangle, Plus, Scissors, X } from 'lucide-react'

import type { FormState, GroupPatient, SystemPromptChoice, TestType } from '@/lib/create-test-config'
import {
  activePromptTexts,
  buildTestRunRequest,
  buildBenchmarkSuiteConfig,
  extractVariables,
  initialFormState,
  newPatientKey,
  numberOr,
  stepsForType,
  validateForm,
} from '@/lib/create-test-config'
import { useBenchmarks, useCreateSuite, useEditedModels, usePromptsByIds } from '@/lib/hooks'
import { isPatientTestPrompt } from '@/lib/prompt-targets'
import { getSettings } from '@/lib/settings'
import { parseModelSelection } from '@/lib/model-overview'
import { toast } from '@/lib/toast'
import { formatApiError } from '@/lib/utils'
import { ModelSelector } from './ModelSelector'
import { EvaluatorStep } from './create-test/EvaluatorStep'
import { GenerationSettings } from './create-test/GenerationSettings'
import { RoleFields } from './create-test/RoleStep'
import { RunSection } from './create-test/RunSection'
import { SystemPromptPicker } from './create-test/SystemPromptPicker'
import { TestDesignSection } from './create-test/TestDesignSection'
import { INPUT, SMALL_BUTTON, StepCard } from './create-test/ui'

interface SuiteModel {
  key: string
  provider: string
  model: string
}

const libraryIds = (choices: SystemPromptChoice[]) =>
  choices.flatMap((c) => (c.mode === 'library' && c.id ? [c.id] : []))

/**
 * Run the same test against several models. Built from the Create Test pieces:
 * the same test design, the same doctor and evaluator steps, and one set of
 * patient settings shared by every model under test. Group therapy puts every
 * model into one session; benchmarks can be several at once.
 */
export function SuiteForm() {
  const router = useRouter()
  const createSuite = useCreateSuite()
  const { data: benchmarksData } = useBenchmarks()
  const { data: edited = [] } = useEditedModels()
  const [initial] = useState(() => initialFormState(getSettings(), router.query).state)
  const [state, setState] = useState<FormState>(initial)
  const [models, setModels] = useState<SuiteModel[]>(() => {
    const selected = parseModelSelection(router.query.models)
    return (selected.length ? selected : [{ provider: initial.patient.provider, model: initial.patient.model }])
      .map(model => ({ ...model, key: newPatientKey() }))
  })
  const [benchmarks, setBenchmarks] = useState<string[]>(() => initial.benchmark ? [initial.benchmark] : [])
  const [attempted, setAttempted] = useState(false)
  const update = (patch: Partial<FormState>) => setState((s) => ({ ...s, ...patch }))
  const steps = stepsForType(state.testType)
  const isBenchmark = state.testType === 'benchmark'
  const isGroup = state.testType === 'group_therapy'

  const complete = models.filter((m) => m.provider && m.model.trim())
  // The validation and request builders work on a single-test form; the suite's
  // models stand in as the patient (or the group's patients) with shared settings.
  const asForm: FormState = {
    ...state,
    patient: { ...state.patient, provider: complete[0]?.provider || '', model: complete[0]?.model || '' },
    groupPatients: complete.map((m): GroupPatient => ({
      key: m.key, provider: m.provider, model: m.model,
      systemPrompt: state.patient.systemPrompt, generation: state.patient.generation,
    })),
  }

  const ids = useMemo(() => Array.from(new Set([
    ...(state.testType === 'one_shot' && state.oneShotSource === 'library' && state.promptId ? [state.promptId] : []),
    ...(state.testType === 'multi_shot' && state.multiShotSource === 'library' ? state.promptIds : []),
    ...libraryIds([state.patient.systemPrompt, state.doctor.systemPrompt, state.evaluator.systemPrompt]),
  ])), [state])
  const testPromptIds = state.testType === 'one_shot' && state.oneShotSource === 'library' && state.promptId ? [state.promptId]
    : state.testType === 'multi_shot' && state.multiShotSource === 'library' ? state.promptIds : []
  const resolved = usePromptsByIds(ids)
  const libraryTexts: Record<number, string> = {}
  const missingPromptIds: number[] = []
  const invalidTestPromptIds: number[] = []
  const pendingTestPromptIds: number[] = []
  ids.forEach((id, i) => {
    if (testPromptIds.includes(id)) {
      if (resolved[i]?.data && !isPatientTestPrompt(resolved[i].data!)) invalidTestPromptIds.push(id)
      else if (!resolved[i]?.data && !resolved[i]?.isError) pendingTestPromptIds.push(id)
    }
    if (resolved[i]?.data) libraryTexts[id] = resolved[i].data!.prompt_text
    else if (resolved[i]?.isError) missingPromptIds.push(id)
  })
  const variableNames = extractVariables(activePromptTexts(state, libraryTexts))

  let errors: string[]
  let warnings: string[] = []
  if (isBenchmark) {
    const benchmarkValidation = validateForm({ ...asForm, benchmark: benchmarks[0] || '' }, { missingPromptIds })
    errors = [
      ...benchmarkValidation.errors.filter((error) => !['Choose a benchmark.', 'Patient: choose a provider and model.'].includes(error)),
      ...(benchmarks.length ? [] : ['Test design: choose at least one benchmark.']),
      ...(complete.length ? [] : ['Models: add at least one model.']),
    ]
  } else {
    ({ errors, warnings } = validateForm(asForm, { missingPromptIds, invalidTestPromptIds, pendingTestPromptIds }))
    errors = errors.map((e) => e
      .replace('Patient: choose a provider and model.', 'Models: add at least one model.')
      .replace('Patients: add at least two patient models for group therapy.', 'Models: group therapy needs at least two models.'))
      .filter((e) => !/^Patient \d+: /.test(e))
  }
  if (complete.length !== models.length) warnings = [...warnings, 'Rows without a provider and model are left out.']

  const runCount = isBenchmark ? benchmarks.length * complete.length : isGroup ? (complete.length ? 1 : 0) : complete.length
  const summary = isBenchmark
    ? `${runCount} benchmark run${runCount === 1 ? '' : 's'}: ${benchmarks.length} benchmark${benchmarks.length === 1 ? '' : 's'} × ${complete.length} model${complete.length === 1 ? '' : 's'}.`
    : isGroup
      ? `One group session with ${complete.length} models; the doctor leads and assesses it.`
      : `${runCount} run${runCount === 1 ? '' : 's'}, one per model; the same doctor assesses every model${state.evaluator.enabled ? ', then each run is scored' : ''}.`

  const setTestType = (testType: TestType) => update({ testType })
  const setModel = (key: string, patch: Partial<SuiteModel>) => setModels((list) => list.map((m) => (m.key === key ? { ...m, ...patch } : m)))
  const addModel = (provider = '', model = '') => setModels((list) => [...list, { key: newPatientKey(), provider, model }])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setAttempted(true)
    if (errors.length) {
      toast.error(errors[0])
      return
    }
    if (runCount > 50 && !window.confirm(`This creates ${runCount} runs and may take a long time. Continue?`)) return
    try {
      const request = isBenchmark ? null : buildTestRunRequest(asForm, variableNames)
      const testConfig = request ? { ...request.test_config } : buildBenchmarkSuiteConfig(asForm)
      if (testConfig) {
        delete testConfig.patients // the suite builds the group from its models
        if (isGroup) applySharedPatientPrompt(testConfig, state.patient.systemPrompt)
        if (request?.variables) testConfig.variables = request.variables
      }
      const result = await createSuite.mutateAsync({
        name: state.name.trim() || undefined,
        test_types: [state.testType],
        benchmarks: isBenchmark ? benchmarks : [],
        models: complete.map((m) => ({ provider: m.provider, model: m.model.trim() })),
        test_config: testConfig,
        num_samples: isBenchmark ? numberOr(state.numSamples, 100, { min: 1, max: 10000, int: true }) : undefined,
        doctor: steps.doctor.shown ? { provider: state.doctor.provider, model: state.doctor.model.trim() } : undefined,
      })
      toast.success(`Suite created with ${runCount} run${runCount === 1 ? '' : 's'}.`)
      router.push(`/suite/${result.id}`)
    } catch (error) {
      toast.error(formatApiError(error, 'Could not create the suite.'))
    }
  }

  const benchmarkRows: any[] = benchmarksData?.benchmarks ?? []
  const benchmarkPicker = (
    <div className="space-y-3">
      <span className="text-sm font-medium">Benchmarks ({benchmarks.length} chosen)</span>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {benchmarkRows.map((b) => {
          const unavailable = b.runnable === false
          const checked = benchmarks.includes(b.name)
          return (
            <label key={b.name} className={`flex items-start gap-2 rounded border p-3 text-sm ${unavailable ? 'opacity-60' : 'cursor-pointer'} ${checked ? 'border-primary bg-primary/10' : 'bg-muted/30'}`}>
              <input type="checkbox" className="mt-1 rounded" disabled={unavailable} checked={checked}
                onChange={() => setBenchmarks((list) => (checked ? list.filter((x) => x !== b.name) : [...list, b.name]))} />
              <span>
                <span className="font-medium">{b.title ?? b.name}</span>
                <span className="block text-xs text-muted-foreground">{b.description}</span>
                {unavailable && <span className="block text-xs text-destructive">{b.unavailable_reason}</span>}
              </span>
            </label>
          )
        })}
      </div>
      <label className="block max-w-xs text-xs font-medium">
        Samples per benchmark
        <input type="number" min="1" max="10000" value={state.numSamples} placeholder="100"
          onChange={(e) => update({ numSamples: e.target.value })} className={INPUT} />
      </label>
      <label className="block max-w-xs text-xs font-medium">
        Seed
        <input type="number" min="0" max="4294967295" value={state.benchmarkSeed} placeholder="0"
          onChange={(e) => update({ benchmarkSeed: e.target.value })} className={INPUT} />
      </label>
    </div>
  )

  let number = 1
  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <TestDesignSection state={state} update={update} setTestType={setTestType} variableNames={variableNames} benchmarkSlot={benchmarkPicker} />

      <StepCard
        number={++number}
        title="Models under test"
        description={isGroup
          ? 'All of these models join one group session as patients.'
          : isBenchmark ? 'Every model answers every chosen benchmark.' : 'Each model gets its own run of the test above.'}
      >
        <div className="space-y-3">
          {models.map((m, index) => (
            <div key={m.key} className="rounded-md border p-4">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-sm font-semibold">Model {index + 1}</h3>
                <button type="button" className={SMALL_BUTTON} disabled={models.length <= 1}
                  onClick={() => setModels((list) => list.filter((x) => x.key !== m.key))} aria-label={`Remove model ${index + 1}`}>
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
              <ModelSelector label="" provider={m.provider} model={m.model}
                onProviderChange={(provider) => setModel(m.key, { provider, model: '' })}
                onModelChange={(model) => setModel(m.key, { model })} />
            </div>
          ))}
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => addModel()}
              className="flex items-center gap-2 rounded-lg border border-dashed px-4 py-2 text-sm font-medium hover:bg-muted">
              <Plus className="h-4 w-4" /> Add model
            </button>
            {Array.isArray(edited) && edited.length > 0 && (
              <>
                <span className="text-xs text-muted-foreground">or add an edited model:</span>
                {edited.slice(0, 4).map((e: any) => (
                  <button key={e.path} type="button" title={e.path} onClick={() => addModel('transformers', e.path)}
                    className="flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm hover:bg-muted">
                    <Scissors className="h-3.5 w-3.5" /> {e.name}
                  </button>
                ))}
              </>
            )}
          </div>
        </div>
          <div className="space-y-4 border-t pt-4">
            <p className="text-xs text-muted-foreground">These patient settings apply to every model above.</p>
            <SystemPromptPicker benchmark={isBenchmark} role="patient" value={state.patient.systemPrompt} allowNone={!isGroup}
              onChange={(systemPrompt) => update({ patient: { ...state.patient, systemPrompt } })} />
            <GenerationSettings defaultTemperature={isBenchmark ? 0 : undefined} defaultMaxTokens={isBenchmark ? 512 : undefined} value={state.patient.generation}
              onChange={(generation) => update({ patient: { ...state.patient, generation } })} />
          </div>
      </StepCard>

      {steps.doctor.shown && (
        <StepCard number={++number} title="Doctor" description={steps.doctor.description}>
          <RoleFields role="doctor" step={state.doctor} onChange={(doctor) => update({ doctor })}
            doctorGoal={steps.doctor.showStrategies ? { value: state.doctorGoal, onChange: (doctorGoal) => update({ doctorGoal }) } : undefined}
            showCot={steps.doctor.showCot} showStrategies={steps.doctor.showStrategies} />
        </StepCard>
      )}

      {steps.evaluator.shown && (
        <EvaluatorStep number={++number} value={state.evaluator} onChange={(evaluator) => update({ evaluator })}
          doctorLabel={state.doctor.model ? `${state.doctor.provider}/${state.doctor.model}` : ''} />
      )}

      <RunSection number={++number} state={asForm} update={update} errors={errors} warnings={warnings}
        summary={summary} attempted={attempted} namePlaceholder="My test suite" />

      {runCount > 50 && (
        <p className="flex items-center gap-2 text-sm text-yellow-700"><AlertTriangle className="h-4 w-4" /> {runCount} runs may take a long time.</p>
      )}
      <div className="flex justify-end">
        <button type="submit" disabled={createSuite.isPending}
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50">
          {createSuite.isPending ? 'Creating…' : `Create suite${runCount ? ` (${runCount} run${runCount === 1 ? '' : 's'})` : ''}`}
        </button>
      </div>
    </form>
  )
}

/** Group therapy: the suite copies the shared patient prompt onto each patient it creates. */
function applySharedPatientPrompt(testConfig: Record<string, any>, choice: SystemPromptChoice) {
  if (choice.mode === 'library') testConfig.patient_system_prompt_id = choice.id
  else if (choice.mode === 'custom' && choice.text.trim()) testConfig.patient_system_prompt = choice.text.trim()
}
