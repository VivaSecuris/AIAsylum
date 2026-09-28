const STORAGE_KEY = 'asylum_settings'

/** Blank numeric fields leave that workflow's runtime default in place. */
export interface GenerationDefaults {
  temperature: string
  top_p: string
  max_tokens: string
  enable_cot: boolean
  use_dynamic_strategies: boolean
}

export type GenerationRole = 'patient' | 'doctor' | 'evaluator'

const EMPTY_GENERATION: GenerationDefaults = {
  temperature: '', top_p: '', max_tokens: '', enable_cot: false, use_dynamic_strategies: true,
}

export interface AppSettings {
  defaultEvaluatorProvider: string
  defaultEvaluatorModel: string
  defaultEnableCotDetection: boolean
  defaultCotAnalysisMode: 'full' | 'partial' | 'none'
  defaultEnableFactualityCheck: boolean
  defaultEnableManipulationAnalysis: boolean
  defaultDoctorProvider: string
  defaultDoctorModel: string
  defaultPatientProvider: string
  defaultPatientModel: string
  defaultDoctorSystemPromptId: number | null
  defaultPatientSystemPromptId: number | null
  defaultEvaluatorSystemPromptId: number | null
  defaultDoctorSystemPromptText: string
  defaultPatientSystemPromptText: string
  defaultEvaluatorSystemPromptText: string
  defaultDoctorGeneration: GenerationDefaults
  defaultPatientGeneration: GenerationDefaults
  defaultEvaluatorGeneration: GenerationDefaults
  defaultAutoAnalysis: boolean
}

export const DEFAULT_SETTINGS: AppSettings = {
  defaultEvaluatorProvider: '',
  defaultEvaluatorModel: '',
  defaultEnableCotDetection: true,
  defaultCotAnalysisMode: 'full',
  defaultEnableFactualityCheck: false,
  defaultEnableManipulationAnalysis: false,
  defaultDoctorProvider: '',
  defaultDoctorModel: '',
  defaultPatientProvider: '',
  defaultPatientModel: '',
  defaultDoctorSystemPromptId: null,
  defaultPatientSystemPromptId: null,
  defaultEvaluatorSystemPromptId: null,
  defaultDoctorSystemPromptText: '',
  defaultPatientSystemPromptText: '',
  defaultEvaluatorSystemPromptText: '',
  defaultDoctorGeneration: { ...EMPTY_GENERATION },
  defaultPatientGeneration: { ...EMPTY_GENERATION },
  defaultEvaluatorGeneration: { ...EMPTY_GENERATION },
  defaultAutoAnalysis: false,
}

export function getRoleGenerationDefaults(settings: Partial<AppSettings>, role: GenerationRole): GenerationDefaults {
  const key = `default${role[0].toUpperCase()}${role.slice(1)}Generation` as
    'defaultPatientGeneration' | 'defaultDoctorGeneration' | 'defaultEvaluatorGeneration'
  const raw = settings[key]
  const out = { ...EMPTY_GENERATION }
  if (!raw || typeof raw !== 'object') return out
  for (const field of ['temperature', 'top_p', 'max_tokens'] as const) {
    if (typeof raw[field] === 'string' || typeof raw[field] === 'number') out[field] = String(raw[field])
  }
  for (const field of ['enable_cot', 'use_dynamic_strategies'] as const) {
    if (typeof raw[field] === 'boolean') out[field] = raw[field]
  }
  return out
}

/** Merge nested defaults too, so pre-generation Settings remain usable. */
export function normalizeSettings(raw: Partial<AppSettings> = {}): AppSettings {
  const settings = { ...DEFAULT_SETTINGS, ...raw }
  settings.defaultPatientGeneration = getRoleGenerationDefaults(raw, 'patient')
  settings.defaultDoctorGeneration = getRoleGenerationDefaults(raw, 'doctor')
  settings.defaultEvaluatorGeneration = getRoleGenerationDefaults(raw, 'evaluator')
  return settings
}

export function getSettings(): AppSettings {
  if (typeof window === 'undefined') return normalizeSettings()
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return normalizeSettings()
    const parsed = JSON.parse(raw)
    return normalizeSettings(parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {})
  } catch {
    return normalizeSettings()
  }
}

export function saveSettings(settings: AppSettings): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(normalizeSettings(settings)))
}
