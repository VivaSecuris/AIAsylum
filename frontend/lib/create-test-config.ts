/**
 * Create Test form state and the request it produces.
 *
 * Pure (no browser or React dependencies) so it can be unit-tested without React. The form is
 * built from Settings, then only the URL keys that are actually present override
 * it; the request is rebuilt from the visible state on every submit, never by
 * spreading a restored config, so a cleared choice is really cleared.
 */
import { DEFAULT_SETTINGS, getRoleGenerationDefaults, normalizeSettings } from './settings'
import type { AppSettings, GenerationDefaults } from './settings'
import type { TestRunRequest } from './api'

export type TestType = 'one_shot' | 'multi_shot' | 'conversation' | 'group_therapy' | 'benchmark'
export type StepRole = 'patient' | 'doctor' | 'evaluator'

export const TEST_TYPES: { value: TestType; label: string; description: string }[] = [
  { value: 'one_shot', label: 'One-Shot', description: 'One prompt, one reply. The doctor then assesses the reply.' },
  { value: 'multi_shot', label: 'Multi-Shot', description: 'Several prompts in one conversation, so context builds up.' },
  { value: 'conversation', label: 'Conversation', description: 'The doctor interviews the patient over several turns.' },
  { value: 'group_therapy', label: 'Group Therapy', description: 'The doctor leads a session with several patient models at once.' },
  { value: 'benchmark', label: 'Benchmark', description: 'Standard dataset questions, scored against the answers.' },
]

/** Where a step's system prompt comes from. `none` (patient only) sends prompts verbatim. */
export type SystemPromptChoice =
  | { mode: 'default' }
  | { mode: 'none' }
  | { mode: 'library'; id: number }
  | { mode: 'custom'; text: string }

/** Kept as strings so a blank field means "use the default". */
export interface GenerationValues extends GenerationDefaults {}

export interface StepState {
  provider: string
  model: string
  systemPrompt: SystemPromptChoice
  generation: GenerationValues
}

export interface GroupPatient extends StepState {
  key: string
}

export interface EvaluatorState {
  enabled: boolean
  provider: string
  model: string
  systemPrompt: SystemPromptChoice
  temperature: string
  top_p: string
  max_tokens: string
  enable_cot: boolean
  enable_cot_detection: boolean
  cot_analysis_mode: 'full' | 'partial' | 'none'
  enable_factuality_check: boolean
  enable_manipulation_analysis: boolean
}

export interface FormState {
  testType: TestType
  oneShotSource: 'library' | 'custom'
  promptId?: number
  customPrompt: string
  multiShotSource: 'library' | 'custom' | 'auto'
  promptIds: number[]
  customPrompts: string
  numMessages: string
  maxTurns: string
  variables: Record<string, string>
  benchmark: string
  numSamples: string
  benchmarkSeed: string
  patient: StepState
  groupPatients: GroupPatient[]
  doctor: StepState
  doctorGoal: string
  evaluator: EvaluatorState
  name: string
  seed: string
  lineageParent?: string
}

export type Query = Record<string, string | string[] | undefined>

export const DEFAULT_TEMPERATURE = 0.7
export const DEFAULT_EVALUATOR_TEMPERATURE = 0.3
export const DEFAULT_MAX_TURNS = 10
export const DEFAULT_NUM_MESSAGES = 10
/** Providers whose integrations reject a generation seed (mirrors `supports_seed` in the backend). */
export const NO_SEED_PROVIDERS = ['anthropic', 'google']

const TYPE_VALUES = TEST_TYPES.map((t) => t.value)

export function emptyGeneration(): GenerationValues {
  return { temperature: '', top_p: '', max_tokens: '', enable_cot: false, use_dynamic_strategies: true }
}

export function libraryChoice(id: number | null | undefined): SystemPromptChoice {
  return id ? { mode: 'library', id } : { mode: 'default' }
}

let keyCounter = 0
export function newPatientKey(): string {
  keyCounter += 1
  return `patient_${keyCounter}_${Math.random().toString(36).slice(2, 7)}`
}

export function newGroupPatient(provider = '', model = '', systemPrompt: SystemPromptChoice = { mode: 'default' },
                                generation: GenerationValues = emptyGeneration()): GroupPatient {
  return { key: newPatientKey(), provider, model, systemPrompt: { ...systemPrompt }, generation: { ...generation } }
}

export function settingsPromptChoice(settings: AppSettings, role: StepRole): SystemPromptChoice {
  const prefix = `default${role[0].toUpperCase()}${role.slice(1)}`
  const id = settings[`${prefix}SystemPromptId` as keyof AppSettings]
  const text = settings[`${prefix}SystemPromptText` as keyof AppSettings]
  return choiceFrom(id, text) || { mode: 'default' }
}

/** Explicit fields override saved defaults, including false, zero, and cleared choices. */
export function initialEvaluatorState(settings: AppSettings, config: Record<string, any> = {}): EvaluatorState {
  const generation = getRoleGenerationDefaults(settings, 'evaluator')
  const hasPrompt = 'evaluator_system_prompt_id' in config || 'evaluator_system_prompt' in config
  const mode = config.cot_analysis_mode ?? settings.defaultCotAnalysisMode
  return {
    enabled: settings.defaultAutoAnalysis ?? false,
    provider: config.evaluator_provider ?? settings.defaultEvaluatorProvider,
    model: config.evaluator_model ?? settings.defaultEvaluatorModel,
    systemPrompt: hasPrompt
      ? choiceFrom(config.evaluator_system_prompt_id, config.evaluator_system_prompt) || { mode: 'default' }
      : settingsPromptChoice(settings, 'evaluator'),
    temperature: String(config.evaluator_temperature ?? generation.temperature),
    top_p: String(config.evaluator_top_p ?? generation.top_p),
    max_tokens: String(config.evaluator_max_tokens ?? generation.max_tokens),
    enable_cot: config.evaluator_enable_cot ?? generation.enable_cot,
    enable_cot_detection: mode === 'none' ? false : config.enable_cot_detection ?? settings.defaultEnableCotDetection,
    cot_analysis_mode: mode === 'none' ? 'full' : mode,
    enable_factuality_check: config.enable_factuality_check ?? settings.defaultEnableFactualityCheck,
    enable_manipulation_analysis: config.enable_manipulation_analysis ?? settings.defaultEnableManipulationAnalysis,
  }
}

export function buildAnalysisConfig(e: EvaluatorState): Record<string, any> {
  const analysis: Record<string, any> = {
    enable_cot_detection: e.enable_cot_detection,
    cot_analysis_mode: e.cot_analysis_mode,
    enable_factuality_check: e.enable_factuality_check,
    enable_manipulation_analysis: e.enable_manipulation_analysis,
    evaluator_enable_cot: e.enable_cot,
    evaluator_temperature: numberOr(e.temperature, DEFAULT_EVALUATOR_TEMPERATURE, { min: 0, max: 2 }),
  }
  if (e.provider && e.model.trim()) {
    analysis.evaluator_provider = e.provider
    analysis.evaluator_model = e.model.trim()
  }
  applyChoice(analysis, 'evaluator_system_prompt_id', 'evaluator_system_prompt', e.systemPrompt)
  const maxTokens = numberOr(e.max_tokens, NaN, { min: 64, max: 32768, int: true })
  if (Number.isFinite(maxTokens)) analysis.evaluator_max_tokens = maxTokens
  const topP = numberOr(e.top_p, NaN, { min: 0, max: 1 })
  if (Number.isFinite(topP) && topP > 0) analysis.evaluator_top_p = topP
  return analysis
}

/** Parse a numeric text field: blank or invalid falls back, out-of-range is clamped out (returns fallback). */
export function numberOr(value: string | number | undefined | null, fallback: number,
                         opts: { min?: number; max?: number; int?: boolean } = {}): number {
  if (value === undefined || value === null || String(value).trim() === '') return fallback
  const parsed = opts.int ? Number.parseInt(String(value), 10) : Number(value)
  if (!Number.isFinite(parsed)) return fallback
  if (opts.min !== undefined && parsed < opts.min) return fallback
  if (opts.max !== undefined && parsed > opts.max) return fallback
  return parsed
}

const first = (value: string | string[] | undefined): string | undefined =>
  Array.isArray(value) ? value[0] : value

export function initialFormState(settings: AppSettings, query: Query = {}): { state: FormState; notices: string[] } {
  settings = normalizeSettings(settings)
  const notices: string[] = []
  const state: FormState = {
    testType: 'one_shot',
    oneShotSource: 'library',
    customPrompt: '',
    multiShotSource: 'library',
    promptIds: [],
    customPrompts: '',
    numMessages: String(DEFAULT_NUM_MESSAGES),
    maxTurns: String(DEFAULT_MAX_TURNS),
    variables: {},
    benchmark: '',
    numSamples: '100',
    benchmarkSeed: '0',
    patient: {
      provider: settings.defaultPatientProvider || 'transformers',
      model: settings.defaultPatientModel || '',
      systemPrompt: settingsPromptChoice(settings, 'patient'),
      generation: getRoleGenerationDefaults(settings, 'patient'),
    },
    groupPatients: [],
    doctor: {
      provider: settings.defaultDoctorProvider || '',
      model: settings.defaultDoctorModel || '',
      systemPrompt: settingsPromptChoice(settings, 'doctor'),
      generation: getRoleGenerationDefaults(settings, 'doctor'),
    },
    evaluator: initialEvaluatorState(settings),
    doctorGoal: '',
    name: '',
    seed: '',
  }
  // "None" as an analysis depth means no chain-of-thought detection at all.
  if (state.evaluator.cot_analysis_mode === 'none') {
    state.evaluator.enable_cot_detection = false
    state.evaluator.cot_analysis_mode = 'full'
  }

  const type = first(query.type)
  if (type) {
    if ((TYPE_VALUES as string[]).includes(type)) state.testType = type as TestType
    else notices.push(`Test type "${type}" is no longer offered; showing One-Shot instead.`)
  }
  const model = first(query.model)
  if (model) {
    state.patient.model = model
    state.patient.provider = first(query.provider) || 'transformers'
  }
  const doctorModel = first(query.doctor_model)
  const doctorProvider = first(query.doctor_provider)
  let config: Record<string, any> | undefined
  const rawConfig = first(query.test_config)
  if (rawConfig) {
    try {
      const parsed = JSON.parse(rawConfig)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) config = parsed
    } catch {
      notices.push('The saved test settings in the link could not be read; defaults are shown instead.')
    }
  }
  // Older one-shot and multi-shot runs recorded the model under test as the doctor too.
  const legacySelfAssessed = !!config && !config.config_version
    && (state.testType === 'one_shot' || state.testType === 'multi_shot')
    && doctorModel === model && (doctorProvider || 'transformers') === (first(query.provider) || 'transformers')
  if (doctorModel && !legacySelfAssessed) {
    state.doctor.model = doctorModel
    state.doctor.provider = doctorProvider || state.doctor.provider
  } else if (legacySelfAssessed) {
    notices.push('The original run used the model under test as its own doctor. The doctor now comes from your Settings; change it below if you want to reproduce the original exactly.')
  }
  if (config) restoreFromTestConfig(state, config)
  const promptId = Number.parseInt(first(query.promptId) || '', 10)
  if (Number.isFinite(promptId) && promptId > 0) {
    if (state.testType === 'multi_shot') {
      state.multiShotSource = 'library'
      state.promptIds = [promptId]
    } else {
      if (state.testType !== 'one_shot') state.testType = 'one_shot'
      state.oneShotSource = 'library'
      state.promptId = promptId
    }
  }
  const lineage = first(query.lineage_parent)
  if (lineage) state.lineageParent = lineage
  return { state, notices }
}

function choiceFrom(id: unknown, text: unknown): SystemPromptChoice | undefined {
  if (typeof id === 'number' && id > 0) return { mode: 'library', id }
  if (typeof id === 'string' && /^\d+$/.test(id)) return { mode: 'library', id: Number(id) }
  if (typeof text === 'string' && text.trim()) return { mode: 'custom', text }
  return undefined
}

function generationFrom(block: any, fallbackTemperature: unknown, fallbackCot: unknown, defaults = emptyGeneration()): GenerationValues {
  const values = { ...defaults }
  const b = block && typeof block === 'object' ? block : {}
  const temperature = b.temperature ?? fallbackTemperature
  if (temperature !== undefined && temperature !== null) values.temperature = String(temperature)
  if (b.top_p !== undefined && b.top_p !== null) values.top_p = String(b.top_p)
  if (b.max_tokens !== undefined && b.max_tokens !== null) values.max_tokens = String(b.max_tokens)
  values.enable_cot = Boolean(b.enable_cot ?? fallbackCot ?? values.enable_cot)
  if (b.use_dynamic_strategies !== undefined) values.use_dynamic_strategies = Boolean(b.use_dynamic_strategies)
  return values
}

/**
 * Restore a saved config. V2 is a complete request: omitted prompt IDs/text mean
 * built-in instructions, and an omitted evaluator model means use the doctor.
 * Legacy partial configs keep the Settings overlay used by older links.
 */
export function restoreFromTestConfig(state: FormState, cfg: Record<string, any>): FormState {
  if (Number(cfg.config_version) >= 2) {
    state.doctorGoal = ''
    state.doctor.systemPrompt = { mode: 'default' }
    state.patient.systemPrompt = { mode: 'default' }
    state.evaluator.systemPrompt = { mode: 'default' }
    state.evaluator.provider = ''
    state.evaluator.model = ''
    state.patient.generation = emptyGeneration()
    state.doctor.generation = emptyGeneration()
    state.evaluator = initialEvaluatorState(DEFAULT_SETTINGS)
  }
  if (typeof cfg.prompt_id === 'number') {
    state.oneShotSource = 'library'
    state.promptId = cfg.prompt_id
  } else if (typeof cfg.prompt === 'string' && cfg.prompt.trim()) {
    state.oneShotSource = 'custom'
    state.customPrompt = cfg.prompt
    state.promptId = undefined
  }
  if (Array.isArray(cfg.prompt_ids) && cfg.prompt_ids.length) {
    state.multiShotSource = 'library'
    state.promptIds = cfg.prompt_ids.filter((id: unknown) => typeof id === 'number')
  } else if (Array.isArray(cfg.prompts) && cfg.prompts.length) {
    state.multiShotSource = 'custom'
    state.customPrompts = cfg.prompts.join('\n')
    if (state.testType === 'one_shot' && !cfg.prompt && cfg.prompts.length === 1) {
      state.oneShotSource = 'custom'
      state.customPrompt = cfg.prompts[0]
    }
  } else if (cfg.num_messages !== undefined && state.testType === 'multi_shot' && typeof cfg.prompt_id !== 'number') {
    state.multiShotSource = 'auto'
  }
  if (cfg.num_messages !== undefined && cfg.num_messages !== null) state.numMessages = String(cfg.num_messages)
  if (cfg.max_turns !== undefined && cfg.max_turns !== null) state.maxTurns = String(cfg.max_turns)
  if (typeof cfg.doctor_goal === 'string') state.doctorGoal = cfg.doctor_goal
  if (cfg.variables && typeof cfg.variables === 'object') state.variables = { ...cfg.variables }

  const doctorPrompt = choiceFrom(cfg.doctor_system_prompt_id, cfg.doctor_system_prompt)
  if (doctorPrompt) state.doctor.systemPrompt = doctorPrompt
  else if ('doctor_system_prompt_id' in cfg || 'doctor_system_prompt' in cfg) state.doctor.systemPrompt = { mode: 'default' }
  const patientPrompt = choiceFrom(cfg.patient_system_prompt_id, cfg.patient_system_prompt)
  if (patientPrompt) state.patient.systemPrompt = patientPrompt
  else if (cfg.patient_prompt_framing === false) state.patient.systemPrompt = { mode: 'none' }
  else if ('patient_system_prompt_id' in cfg || 'patient_system_prompt' in cfg) state.patient.systemPrompt = { mode: 'default' }

  const roles = cfg.roles && typeof cfg.roles === 'object' ? cfg.roles : {}
  if (roles.doctor || cfg.temperature !== undefined || cfg.enable_doctor_cot !== undefined || cfg.use_dynamic_strategies !== undefined) {
    state.doctor.generation = generationFrom(roles.doctor, cfg.temperature, cfg.enable_doctor_cot, state.doctor.generation)
    if (!roles.doctor && cfg.use_dynamic_strategies !== undefined) {
      state.doctor.generation.use_dynamic_strategies = Boolean(cfg.use_dynamic_strategies)
    }
  }
  if (roles.patient || cfg.temperature !== undefined || cfg.enable_patient_cot !== undefined) {
    state.patient.generation = generationFrom(roles.patient, cfg.temperature, cfg.enable_patient_cot, state.patient.generation)
  }
  if (Array.isArray(cfg.patients)) {
    state.groupPatients = cfg.patients
      .filter((p: any) => p && typeof p === 'object')
      .map((p: any) => ({
        key: newPatientKey(),
        provider: p.provider || '',
        model: p.model || '',
        systemPrompt: choiceFrom(p.system_prompt_id, p.system_prompt) || { ...state.patient.systemPrompt },
        generation: p.generation ? generationFrom(p.generation, undefined, undefined) : { ...state.patient.generation },
      }))
  }
  if (cfg.seed !== undefined && cfg.seed !== null && cfg.seed !== '') state.seed = String(cfg.seed)

  if (cfg.auto_analysis !== undefined) state.evaluator.enabled = Boolean(cfg.auto_analysis)
  if (cfg.auto_analysis || cfg.analysis_config) {
    const a = cfg.analysis_config && typeof cfg.analysis_config === 'object' ? cfg.analysis_config : {}
    if (a.evaluator_provider) state.evaluator.provider = a.evaluator_provider
    if (a.evaluator_model) state.evaluator.model = a.evaluator_model
    const evaluatorPrompt = choiceFrom(a.evaluator_system_prompt_id, a.evaluator_system_prompt)
    if (evaluatorPrompt) state.evaluator.systemPrompt = evaluatorPrompt
    else if ('evaluator_system_prompt_id' in a || 'evaluator_system_prompt' in a) state.evaluator.systemPrompt = { mode: 'default' }
    if (a.evaluator_temperature !== undefined && a.evaluator_temperature !== null) state.evaluator.temperature = String(a.evaluator_temperature)
    if (a.evaluator_enable_cot !== undefined) state.evaluator.enable_cot = Boolean(a.evaluator_enable_cot)
    if (a.evaluator_top_p !== undefined && a.evaluator_top_p !== null) state.evaluator.top_p = String(a.evaluator_top_p)
    if (a.evaluator_max_tokens !== undefined && a.evaluator_max_tokens !== null) state.evaluator.max_tokens = String(a.evaluator_max_tokens)
    for (const key of ['enable_cot_detection', 'enable_factuality_check', 'enable_manipulation_analysis'] as const) {
      if (typeof a[key] === 'boolean') state.evaluator[key] = a[key]
    }
    if (a.cot_analysis_mode === 'none') state.evaluator.enable_cot_detection = false
    else if (['full', 'partial'].includes(a.cot_analysis_mode)) state.evaluator.cot_analysis_mode = a.cot_analysis_mode
  }

  const benchmark = cfg.benchmark_name || cfg.benchmark
  if (typeof benchmark === 'string') state.benchmark = benchmark
  if (cfg.num_samples !== undefined) state.numSamples = String(cfg.num_samples)
  if (state.testType === 'benchmark' && cfg.seed !== undefined && cfg.seed !== null) state.benchmarkSeed = String(cfg.seed)
  if (state.testType === 'benchmark' && (roles.patient || cfg.temperature !== undefined || cfg.top_p !== undefined || cfg.enable_cot !== undefined || cfg.max_new_tokens !== undefined)) {
    state.patient.generation = generationFrom(roles.patient || {
      top_p: cfg.top_p, max_tokens: cfg.max_new_tokens,
    }, cfg.temperature, cfg.enable_cot, state.patient.generation)
  }
  if (cfg.max_new_tokens !== undefined) {
    state.patient.generation.max_tokens = String(cfg.max_new_tokens)
  }
  return state
}

/** What each step does for a test type, and which step-specific options apply. */
export function stepsForType(type: TestType) {
  const oneOrMulti = type === 'one_shot' || type === 'multi_shot'
  return {
    patient: {
      shown: true,
      title: type === 'group_therapy' ? 'Patients' : 'Patient',
      description: {
        one_shot: 'The model under test. It gets the test prompt once; its reply is recorded and scored.',
        multi_shot: 'The model under test. It gets each prompt in order in one conversation, so context builds up.',
        conversation: 'The model under test. It answers the doctor’s questions each turn.',
        group_therapy: 'The models under test. Each hears the doctor and the other patients, then answers.',
        benchmark: 'The model under test. Dataset questions keep their standard answer format; selected system and generation settings are recorded as part of the test protocol.',
      }[type],
      showSystemPrompt: true,
      showGeneration: true,
    },
    doctor: {
      shown: type !== 'benchmark',
      description: oneOrMulti
        ? 'Never talks to the patient. Reads the transcript afterwards and writes the clinical assessment.'
        : type === 'conversation'
          ? 'Interviews the patient for up to the set number of turns, then writes the assessment.'
          : 'Leads the group each turn, then writes the assessment.',
      showCot: type !== 'benchmark',
      showStrategies: type === 'conversation' || type === 'group_therapy',
    },
    evaluator: {
      shown: type !== 'benchmark',
      description: 'Scores the transcript on the eight safety dimensions after the test finishes.',
    },
    seed: type !== 'benchmark',
  }
}

function generationPayload(g: GenerationValues, role: 'doctor' | 'patient', withStrategies: boolean): Record<string, any> {
  const out: Record<string, any> = {
    temperature: numberOr(g.temperature, DEFAULT_TEMPERATURE, { min: 0, max: 2 }),
    enable_cot: g.enable_cot,
  }
  const topP = numberOr(g.top_p, NaN, { min: 0, max: 1 })
  if (Number.isFinite(topP) && topP > 0) out.top_p = topP
  const maxTokens = numberOr(g.max_tokens, NaN, { min: 1, max: 32768, int: true })
  if (Number.isFinite(maxTokens)) out.max_tokens = maxTokens
  if (role === 'doctor' && withStrategies) out.use_dynamic_strategies = g.use_dynamic_strategies
  return out
}

function applyChoice(target: Record<string, any>, idKey: string, textKey: string, choice: SystemPromptChoice) {
  if (choice.mode === 'library') target[idKey] = choice.id
  else if (choice.mode === 'custom' && choice.text.trim()) target[textKey] = choice.text
}

export function splitLines(text: string): string[] {
  return text.split('\n').map((line) => line.trim()).filter(Boolean)
}

/** Which prompt texts are in play, for variable extraction. Library prompt texts come from `libraryTexts`. */
export function activePromptTexts(state: FormState, libraryTexts: Record<number, string>): string[] {
  if (state.testType === 'one_shot') {
    if (state.oneShotSource === 'custom') return [state.customPrompt]
    return state.promptId && libraryTexts[state.promptId] ? [libraryTexts[state.promptId]] : []
  }
  if (state.testType === 'multi_shot') {
    if (state.multiShotSource === 'custom') return splitLines(state.customPrompts)
    if (state.multiShotSource === 'library') return state.promptIds.map((id) => libraryTexts[id] || '')
  }
  return []
}

export function extractVariables(texts: string[]): string[] {
  const names = new Set<string>()
  for (const text of texts) {
    Array.from(text.matchAll(/\$([A-Za-z_][A-Za-z0-9_]*)/g), (match) => names.add(match[1]))
  }
  return Array.from(names).sort()
}

/** The request for a non-benchmark run, built only from what the form shows. */
export function buildTestRunRequest(state: FormState, variableNames: string[] = []): TestRunRequest {
  const steps = stepsForType(state.testType)
  const cfg: Record<string, any> = { config_version: 2 }
  if (steps.doctor.showStrategies && state.doctorGoal?.trim()) cfg.doctor_goal = state.doctorGoal
  if (state.testType === 'one_shot') {
    if (state.oneShotSource === 'library' && state.promptId) cfg.prompt_id = state.promptId
    else if (state.oneShotSource === 'custom') cfg.prompt = state.customPrompt.trim()
  } else if (state.testType === 'multi_shot') {
    if (state.multiShotSource === 'library') cfg.prompt_ids = [...state.promptIds]
    else if (state.multiShotSource === 'custom') cfg.prompts = splitLines(state.customPrompts)
    else cfg.num_messages = numberOr(state.numMessages, DEFAULT_NUM_MESSAGES, { min: 1, max: 100, int: true })
  } else {
    cfg.max_turns = numberOr(state.maxTurns, DEFAULT_MAX_TURNS, { min: 1, max: 50, int: true })
  }

  applyChoice(cfg, 'doctor_system_prompt_id', 'doctor_system_prompt', state.doctor.systemPrompt)
  cfg.roles = {
    doctor: generationPayload(state.doctor.generation, 'doctor', steps.doctor.showStrategies),
    patient: generationPayload(state.patient.generation, 'patient', false),
  }

  let patientProvider = state.patient.provider
  let patientModel = state.patient.model
  if (state.testType === 'group_therapy') {
    const patients = state.groupPatients.filter((p) => p.provider && p.model.trim())
    cfg.patients = patients.map((p) => {
      const entry: Record<string, any> = { provider: p.provider, model: p.model.trim(),
        generation: generationPayload(p.generation, 'patient', false) }
      applyChoice(entry, 'system_prompt_id', 'system_prompt', p.systemPrompt)
      return entry
    })
    if (patients.length) {
      patientProvider = patients[0].provider
      patientModel = patients[0].model.trim()
    }
  } else {
    applyChoice(cfg, 'patient_system_prompt_id', 'patient_system_prompt', state.patient.systemPrompt)
    if (state.patient.systemPrompt.mode === 'none') cfg.patient_prompt_framing = false
  }

  const seed = numberOr(state.seed, NaN, { min: 0, max: 4294967295, int: true })
  if (Number.isFinite(seed)) cfg.seed = seed

  if (state.evaluator.enabled) {
    const analysis = buildAnalysisConfig(state.evaluator)
    cfg.auto_analysis = true
    cfg.analysis_config = analysis
  }

  const variables: Record<string, string> = {}
  for (const name of variableNames) {
    if (state.variables[name]) variables[name] = state.variables[name]
  }
  return {
    doctor_provider: state.doctor.provider,
    doctor_model: state.doctor.model.trim(),
    patient_provider: patientProvider,
    patient_model: patientModel.trim(),
    test_type: state.testType,
    test_config: cfg,
    ...(Object.keys(variables).length ? { variables } : {}),
    ...(state.name.trim() ? { name: state.name.trim() } : {}),
    ...(state.lineageParent ? { lineage_parent: state.lineageParent } : {}),
  }
}

export function buildBenchmarkRequest(state: FormState) {
  const prompt: { patient_system_prompt_id?: number; patient_system_prompt?: string; patient_prompt_framing?: boolean } = {}
  applyChoice(prompt, 'patient_system_prompt_id', 'patient_system_prompt', state.patient.systemPrompt)
  if (state.patient.systemPrompt.mode === 'none') prompt.patient_prompt_framing = false
  const g = state.patient.generation
  const topP = numberOr(g.top_p, NaN, { min: 0, max: 1 })
  return {
    provider: state.patient.provider,
    model: state.patient.model.trim(),
    benchmark: state.benchmark,
    ...prompt,
    num_samples: numberOr(state.numSamples, 100, { min: 1, max: 10000, int: true }),
    seed: numberOr(state.benchmarkSeed, 0, { min: 0, max: 4294967295, int: true }),
    max_new_tokens: numberOr(g.max_tokens, 512, { min: 1, max: 32768, int: true }),
    temperature: numberOr(g.temperature, 0, { min: 0, max: 2 }),
    ...(Number.isFinite(topP) && topP > 0 ? { top_p: topP } : {}),
    enable_cot: g.enable_cot,
  }
}

/** The suite runner reads benchmark options from its shared test_config. */
export function buildBenchmarkSuiteConfig(state: FormState): Record<string, any> {
  const { provider, model, benchmark, num_samples, ...options } = buildBenchmarkRequest(state)
  return { config_version: 2, ...options, roles: { patient: { temperature: options.temperature, enable_cot: options.enable_cot,
    max_tokens: options.max_new_tokens, ...(options.top_p !== undefined ? { top_p: options.top_p } : {}) } } }
}

/** Which steps would receive the run seed. */
export function seedTargets(state: FormState): { applied: string[]; skipped: string[] } {
  const steps: [string, string][] = []
  if (state.testType === 'group_therapy') {
    state.groupPatients.forEach((p, i) => steps.push([`Patient ${i + 1}`, p.provider]))
  } else {
    steps.push(['Patient', state.patient.provider])
  }
  if (stepsForType(state.testType).doctor.shown) steps.push(['Doctor', state.doctor.provider])
  const applied: string[] = []
  const skipped: string[] = []
  for (const [label, provider] of steps) {
    if (NO_SEED_PROVIDERS.includes(provider)) skipped.push(`${label} (${provider})`)
    else applied.push(label)
  }
  return { applied, skipped }
}

export interface ValidationContext {
  /** Library prompt IDs chosen in the form that no longer exist. */
  missingPromptIds?: number[]
  invalidTestPromptIds?: number[]
  pendingTestPromptIds?: number[]
  unfilledVariables?: string[]
}

function checkGeneration(label: string, g: GenerationValues, errors: string[]) {
  const check = (value: string, name: string, ok: (n: number) => boolean, range: string) => {
    if (value.trim() === '') return
    const n = Number(value)
    if (!Number.isFinite(n) || !ok(n)) errors.push(`${label}: ${name} must be ${range}.`)
  }
  check(g.temperature, 'temperature', (n) => n >= 0 && n <= 2, 'between 0 and 2')
  check(g.top_p, 'top-p', (n) => n > 0 && n <= 1, 'above 0 and at most 1')
  check(g.max_tokens, 'max tokens', (n) => Number.isInteger(n) && n >= 1 && n <= 32768, 'a whole number from 1 to 32768')
}

function checkChoice(label: string, choice: SystemPromptChoice, missing: number[], errors: string[]) {
  if (choice.mode === 'custom' && !choice.text.trim()) errors.push(`${label}: the custom system prompt is empty.`)
  if (choice.mode === 'library' && choice.id <= 0) errors.push(`${label}: choose a library system prompt.`)
  if (choice.mode === 'library' && missing.includes(choice.id)) {
    errors.push(`${label}: system prompt #${choice.id} no longer exists. Choose another one.`)
  }
}

export function validateForm(state: FormState, ctx: ValidationContext = {}): { errors: string[]; warnings: string[] } {
  const errors: string[] = []
  const warnings: string[] = []
  const missing = ctx.missingPromptIds || []
  const steps = stepsForType(state.testType)

  if (state.testType === 'benchmark') {
    if (!state.benchmark) errors.push('Choose a benchmark.')
    if (!state.patient.provider || !state.patient.model.trim()) errors.push('Patient: choose a provider and model.')
    checkGeneration('Patient', state.patient.generation, errors)
    checkChoice('Patient', state.patient.systemPrompt, missing, errors)
    return { errors, warnings }
  }

  if (steps.doctor.showStrategies && Array.from(state.doctorGoal || '').length > 8000) {
    errors.push('Doctor: the conversation goal must be at most 8,000 characters.')
  }

  const selectedTestIds = state.testType === 'one_shot' && state.oneShotSource === 'library' && state.promptId ? [state.promptId]
    : state.testType === 'multi_shot' && state.multiShotSource === 'library' ? state.promptIds : []
  if (selectedTestIds.some((id) => ctx.invalidTestPromptIds?.includes(id))) {
    errors.push('Test design: choose patient or general test prompts. Doctor and evaluator prompts belong to their own roles.')
  }
  if (selectedTestIds.some((id) => ctx.pendingTestPromptIds?.includes(id))) {
    errors.push('Test design: wait for the selected prompts to be checked.')
  }

  if (state.testType === 'one_shot') {
    if (state.oneShotSource === 'library') {
      if (!state.promptId) errors.push('Test design: choose a test prompt, or write a custom one.')
      else if (missing.includes(state.promptId)) errors.push(`Test design: prompt #${state.promptId} no longer exists.`)
    } else if (!state.customPrompt.trim()) {
      errors.push('Test design: the custom prompt is empty.')
    }
  } else if (state.testType === 'multi_shot') {
    if (state.multiShotSource === 'library' && state.promptIds.length === 0) errors.push('Test design: add at least one prompt.')
    if (state.multiShotSource === 'custom' && splitLines(state.customPrompts).length === 0) errors.push('Test design: write at least one prompt line.')
    if (state.multiShotSource === 'auto' && !Number.isFinite(numberOr(state.numMessages, NaN, { min: 1, max: 100, int: true }))) {
      errors.push('Test design: number of messages must be from 1 to 100.')
    }
    const gone = state.promptIds.filter((id) => missing.includes(id))
    if (state.multiShotSource === 'library' && gone.length) errors.push(`Test design: prompt ${gone.map((id) => `#${id}`).join(', ')} no longer exists.`)
  } else if (state.maxTurns.trim() !== '' && !Number.isFinite(numberOr(state.maxTurns, NaN, { min: 1, max: 50, int: true }))) {
    errors.push('Test design: max turns must be from 1 to 50.')
  }

  if (state.testType === 'group_therapy') {
    const complete = state.groupPatients.filter((p) => p.provider && p.model.trim())
    if (complete.length < 2) errors.push('Patients: add at least two patient models for group therapy.')
    if (complete.length !== state.groupPatients.length) warnings.push('Patients without a provider and model are left out.')
    state.groupPatients.forEach((p, i) => {
      checkChoice(`Patient ${i + 1}`, p.systemPrompt, missing, errors)
      checkGeneration(`Patient ${i + 1}`, p.generation, errors)
    })
  } else {
    if (!state.patient.provider || !state.patient.model.trim()) errors.push('Patient: choose a provider and model.')
    checkChoice('Patient', state.patient.systemPrompt, missing, errors)
    checkGeneration('Patient', state.patient.generation, errors)
  }

  if (steps.doctor.shown) {
    if (!state.doctor.provider || !state.doctor.model.trim()) errors.push('Doctor: choose a provider and model.')
    checkChoice('Doctor', state.doctor.systemPrompt, missing, errors)
    checkGeneration('Doctor', state.doctor.generation, errors)
  }

  if (state.evaluator.enabled) {
    const e = state.evaluator
    if (!!e.provider !== !!e.model.trim()) errors.push('Evaluator: choose both a provider and a model, or leave both blank to use the doctor.')
    checkChoice('Evaluator', e.systemPrompt, missing, errors)
    if (e.temperature.trim() !== '' && !Number.isFinite(numberOr(e.temperature, NaN, { min: 0, max: 2 }))) {
      errors.push('Evaluator: temperature must be between 0 and 2.')
    }
    if (e.top_p.trim() !== '' && !Number.isFinite(numberOr(e.top_p, NaN, { min: Number.MIN_VALUE, max: 1 }))) {
      errors.push('Evaluator: top-p must be above 0 and at most 1.')
    }
    if (e.max_tokens.trim() !== '' && !Number.isFinite(numberOr(e.max_tokens, NaN, { min: 512, max: 32768, int: true }))) {
      errors.push('Evaluator: max tokens must be a whole number from 512 to 32768 (the scores are a long JSON answer).')
    }
  }

  if (state.seed.trim() !== '' && !Number.isFinite(numberOr(state.seed, NaN, { min: 0, max: 4294967295, int: true }))) {
    errors.push('Run: seed must be a whole number from 0 to 4294967295.')
  }
  if (ctx.unfilledVariables?.length) {
    warnings.push(`Variables left blank are sent as written: ${ctx.unfilledVariables.map((v) => `$${v}`).join(', ')}.`)
  }
  return { errors, warnings }
}
