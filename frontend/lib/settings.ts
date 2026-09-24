const STORAGE_KEY = 'asylum_settings'

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
}

export function getSettings(): AppSettings {
  if (typeof window === 'undefined') return { ...DEFAULT_SETTINGS }
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULT_SETTINGS }
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) }
  } catch {
    return { ...DEFAULT_SETTINGS }
  }
}

export function saveSettings(settings: AppSettings): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
}
