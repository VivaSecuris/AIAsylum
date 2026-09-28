import { useMemo, useState } from 'react'
import { useRouter } from 'next/router'

import type { FormState, GroupPatient, Query, StepState, SystemPromptChoice, TestType } from '@/lib/create-test-config'
import {
  TEST_TYPES,
  activePromptTexts,
  buildBenchmarkRequest,
  buildTestRunRequest,
  extractVariables,
  initialFormState,
  newGroupPatient,
  splitLines,
  stepsForType,
  validateForm,
} from '@/lib/create-test-config'
import { useCreateTestRun, usePromptsByIds, useRunBenchmark } from '@/lib/hooks'
import { isPatientTestPrompt } from '@/lib/prompt-targets'
import { getSettings } from '@/lib/settings'
import { toast } from '@/lib/toast'
import { formatApiError } from '@/lib/utils'
import { EvaluatorStep } from './EvaluatorStep'
import { GroupPatientsStep } from './GroupPatientsStep'
import { RoleFields } from './RoleStep'
import { RunSection } from './RunSection'
import { TestDesignSection } from './TestDesignSection'
import { SMALL_BUTTON, StepCard } from './ui'

/** "provider/model", with a local checkpoint path shortened to its folder name. */
const modelLabel = (step: { provider: string; model: string }) => {
  const model = step.model.trim()
  if (!step.provider || !model) return ''
  return model.startsWith('/') ? model.split('/').filter(Boolean).pop() || model : `${step.provider}/${model}`
}

const libraryIds = (choices: SystemPromptChoice[]) =>
  choices.flatMap((c) => (c.mode === 'library' && c.id ? [c.id] : []))

/**
 * Create a test run, one card per step in the order they run: the test design,
 * the patient (model under test), the doctor, the evaluator, and the run itself.
 * Mounted only on the client (see pages/create-test.tsx), so Settings and the URL
 * are read once, before the first render.
 */
export function CreateTestForm({ query }: { query: Query }) {
  const router = useRouter()
  const createTestRun = useCreateTestRun()
  const runBenchmark = useRunBenchmark()
  const [{ state: initial, notices }] = useState(() => initialFormState(getSettings(), query))
  const [state, setState] = useState<FormState>(initial)
  const [attempted, setAttempted] = useState(false)
  const update = (patch: Partial<FormState>) => setState((s) => ({ ...s, ...patch }))
  const steps = stepsForType(state.testType)

  const setTestType = (testType: TestType) => setState((s) => {
    const next = { ...s, testType }
    // Group therapy starts from the chosen patient plus one more slot.
    if (testType === 'group_therapy' && s.groupPatients.length === 0) {
      const promptChoice = s.patient.systemPrompt.mode === 'none' ? { mode: 'default' as const } : s.patient.systemPrompt
      next.groupPatients = [
        { ...newGroupPatient(s.patient.provider, s.patient.model, promptChoice), generation: { ...s.patient.generation } },
        newGroupPatient(s.patient.provider, s.patient.model, promptChoice, s.patient.generation),
      ]
    }
    return next
  })

  // Every library prompt the form refers to, resolved by ID: to show the texts,
  // extract $variables, and catch IDs (often stale Settings defaults) that are gone.
  const ids = useMemo(() => Array.from(new Set([
    ...(state.testType === 'one_shot' && state.oneShotSource === 'library' && state.promptId ? [state.promptId] : []),
    ...(state.testType === 'multi_shot' && state.multiShotSource === 'library' ? state.promptIds : []),
    ...libraryIds([state.patient.systemPrompt, state.doctor.systemPrompt, state.evaluator.systemPrompt]),
    ...libraryIds(state.groupPatients.map((p) => p.systemPrompt)),
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
    const q = resolved[i]
    if (q?.data) libraryTexts[id] = q.data.prompt_text
    else if (q?.isError) missingPromptIds.push(id)
  })

  const variableNames = extractVariables(activePromptTexts(state, libraryTexts))
  const { errors, warnings } = validateForm(state, {
    missingPromptIds, invalidTestPromptIds, pendingTestPromptIds,
    unfilledVariables: variableNames.filter((name) => !state.variables[name]?.trim()),
  })

  const settings = getSettings()
  const settingsDoctor = { provider: settings.defaultDoctorProvider, model: settings.defaultDoctorModel }
  const doctorName = modelLabel(state.doctor)
  const patientName = state.testType === 'group_therapy'
    ? `${state.groupPatients.filter((p) => modelLabel(p)).length} patients`
    : modelLabel(state.patient) || 'the patient'
  const what = {
    one_shot: state.oneShotSource === 'library' ? 'one library prompt' : 'one custom prompt',
    multi_shot: state.multiShotSource === 'library' ? `${state.promptIds.length} library prompts`
      : state.multiShotSource === 'custom' ? `${splitLines(state.customPrompts).length} custom prompts` : `${state.numMessages || 10} generated messages`,
    conversation: `up to ${state.maxTurns || 10} turns`,
    group_therapy: `up to ${state.maxTurns || 10} turns`,
    benchmark: state.benchmark ? `the ${state.benchmark} benchmark` : 'a benchmark',
  }[state.testType]
  const typeLabel = TEST_TYPES.find((t) => t.value === state.testType)?.label
  const groupSize = state.groupPatients.filter((p) => modelLabel(p)).length
  const scoring = state.evaluator.enabled
    ? `, then ${modelLabel(state.evaluator) || doctorName || 'the doctor'} scores it.` : '. No automatic scoring.'
  const summary = state.testType === 'benchmark'
    ? `${typeLabel}: ${patientName} answers ${what}.`
    : state.testType === 'group_therapy'
      ? `${typeLabel}: ${groupSize} patient${groupSize === 1 ? '' : 's'} in a session of ${what}; ${doctorName || 'the doctor'} leads it and writes the assessment${scoring}`
      : `${typeLabel}: ${patientName} gets ${what}; ${doctorName || 'the doctor'} writes the assessment${scoring}`

  const submitting = createTestRun.isPending || runBenchmark.isPending
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setAttempted(true)
    if (errors.length) {
      toast.error(errors[0])
      return
    }
    try {
      if (state.testType === 'benchmark') {
        const result = await runBenchmark.mutateAsync(buildBenchmarkRequest(state))
        const id = result.test_run_id || result.id
        toast.success(`Benchmark started (run #${id}).`)
        router.push(`/test-runs/${id}`)
        return
      }
      const result = await createTestRun.mutateAsync(buildTestRunRequest(state, variableNames))
      toast.success(state.evaluator.enabled ? 'Test started. Analysis will run when it completes.' : 'Test started.')
      router.push(`/test-runs/${result.id}`)
    } catch (error) {
      toast.error(formatApiError(error, 'Could not create the test run.'))
    }
  }

  const setPatient = (patient: StepState) => update({ patient })
  const setDoctor = (doctor: StepState) => update({ doctor })
  let number = 1

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {notices.length > 0 && (
        <div className="rounded-lg border bg-primary/5 p-4 text-sm">
          {notices.map((notice) => <p key={notice}>{notice}</p>)}
        </div>
      )}

      <TestDesignSection state={state} update={update} setTestType={setTestType} variableNames={variableNames} />

      <StepCard number={++number} title={steps.patient.title} description={steps.patient.description}>
        {state.testType === 'group_therapy' ? (
          <GroupPatientsStep defaults={state.patient} patients={state.groupPatients} onChange={(groupPatients: GroupPatient[]) => update({ groupPatients })} />
        ) : (
          <RoleFields role="patient" step={state.patient} onChange={setPatient} allowNone
            showSystemPrompt={steps.patient.showSystemPrompt} showGeneration={steps.patient.showGeneration}
            benchmark={state.testType === 'benchmark'} defaultTemperature={state.testType === 'benchmark' ? 0 : undefined} defaultMaxTokens={state.testType === 'benchmark' ? 512 : undefined} />
        )}
      </StepCard>

      {steps.doctor.shown && (
        <StepCard
          number={++number}
          title="Doctor"
          description={steps.doctor.description}
          actions={
            <>
              {settingsDoctor.provider && settingsDoctor.model && modelLabel(settingsDoctor) !== doctorName && (
                <button type="button" className={SMALL_BUTTON} onClick={() => setDoctor({ ...state.doctor, ...settingsDoctor })}>
                  Use Settings default
                </button>
              )}
              {state.testType !== 'group_therapy' && modelLabel(state.patient) && modelLabel(state.patient) !== doctorName && (
                <button type="button" className={SMALL_BUTTON}
                  onClick={() => setDoctor({ ...state.doctor, provider: state.patient.provider, model: state.patient.model })}>
                  Use the patient model
                </button>
              )}
            </>
          }
        >
          <RoleFields role="doctor" step={state.doctor} onChange={setDoctor}
            doctorGoal={steps.doctor.showStrategies ? { value: state.doctorGoal, onChange: (doctorGoal) => update({ doctorGoal }) } : undefined}
            showCot={steps.doctor.showCot} showStrategies={steps.doctor.showStrategies} />
          {doctorName && doctorName === modelLabel(state.patient) && (state.testType === 'one_shot' || state.testType === 'multi_shot') && (
            <p className="text-xs text-muted-foreground">The model under test will assess its own replies.</p>
          )}
        </StepCard>
      )}

      {steps.evaluator.shown && (
        <EvaluatorStep number={++number} value={state.evaluator} doctorLabel={doctorName}
          onChange={(evaluator) => update({ evaluator })} />
      )}

      <RunSection number={++number} state={state} update={update} errors={errors} warnings={warnings} summary={summary} attempted={attempted} />

      <div className="flex justify-end gap-3">
        <button type="button" onClick={() => router.back()} className="rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted">
          Cancel
        </button>
        <button type="submit" disabled={submitting}
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50">
          {submitting ? 'Starting…' : state.testType === 'benchmark' ? 'Run benchmark' : 'Create test run'}
        </button>
      </div>
    </form>
  )
}
