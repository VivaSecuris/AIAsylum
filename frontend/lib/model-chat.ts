import type { apiClient } from '@/lib/api'
import type { AppSettings, GenerationRole } from '@/lib/settings'
import { getRoleGenerationDefaults } from '@/lib/settings'
import { buildTestRunRequest, initialFormState } from '@/lib/create-test-config'

export type ChatResponse = Awaited<ReturnType<typeof apiClient.chatWithAnyModel>>
export interface ChatSettings {
  system_prompt: string
  temperature: number | null
  top_p: number | null
  max_tokens: number | null
  seed: number | null
  enable_cot: boolean
  device: 'auto' | 'cpu' | 'cuda' | 'mps'
  dtype: 'float32' | 'float16' | 'bfloat16'
}
export interface ChatExchange {
  prompt: string
  response: ChatResponse
  at: string
}

const characterCount = (text: string) => Array.from(text).length
const optionalNumber = (value: string) => value.trim() ? Number(value) : null

const ROLE_DEFAULT_KEYS = {
  patient: ['defaultPatientSystemPromptId', 'defaultPatientSystemPromptText', 'defaultPatientGeneration'],
  doctor: ['defaultDoctorSystemPromptId', 'defaultDoctorSystemPromptText', 'defaultDoctorGeneration'],
  evaluator: ['defaultEvaluatorSystemPromptId', 'defaultEvaluatorSystemPromptText', 'defaultEvaluatorGeneration'],
} as const

export function chatDefaultPrompt(defaults: AppSettings, role: GenerationRole = 'patient') {
  const [idKey, textKey] = ROLE_DEFAULT_KEYS[role]
  return { id: defaults[idKey], text: defaults[textKey] || '' }
}

export function chatSettingsFromDefaults(defaults: AppSettings, role: GenerationRole = 'patient'): ChatSettings {
  const generation = getRoleGenerationDefaults(defaults, role)
  const prompt = chatDefaultPrompt(defaults, role)
  return {
    system_prompt: prompt.id ? '' : prompt.text,
    temperature: optionalNumber(generation.temperature), top_p: optionalNumber(generation.top_p),
    max_tokens: optionalNumber(generation.max_tokens), seed: null, enable_cot: generation.enable_cot,
    device: 'auto', dtype: 'bfloat16',
  }
}

/** Keep all other saved role/model choices when saving this conversation's defaults. */
export function settingsWithChatDefaults(defaults: AppSettings, settings: ChatSettings, role: GenerationRole = 'patient'): AppSettings {
  validateChatSettings(settings)
  const [idKey, textKey, generationKey] = ROLE_DEFAULT_KEYS[role]
  return {
    ...defaults,
    [idKey]: null,
    [textKey]: settings.system_prompt,
    [generationKey]: {
      ...getRoleGenerationDefaults(defaults, role),
      temperature: settings.temperature == null ? '' : String(settings.temperature),
      top_p: settings.top_p == null ? '' : String(settings.top_p),
      max_tokens: settings.max_tokens == null ? '' : String(settings.max_tokens),
      enable_cot: settings.enable_cot,
    },
  }
}

export function modelChatUrl(provider: string, model: string) {
  return { pathname: '/models/chat', query: { provider: provider || 'transformers', model } }
}

export function supportsWeightActions(provider: string) {
  return ['transformers', 'local'].includes(provider.toLowerCase())
}

/** A completed reply can resolve provider defaults for the next workflow. */
export function chatSettingsForHandoff(settings: ChatSettings, response?: ChatResponse): ChatSettings {
  const applied = response?.generation?.applied || {}
  const resolved = { ...settings }
  for (const field of ['temperature', 'top_p', 'max_tokens', 'seed'] as const) {
    if (resolved[field] == null && typeof applied[field] === 'number' && Number.isFinite(applied[field])) resolved[field] = applied[field]
  }
  return resolved
}

/** Constant local destinations and structured queries preserve IDs containing slashes, + or #. */
export function modelActionUrls(provider: string, model: string, settings: ChatSettings, prompt: string, defaults: AppSettings, sourceModel?: string | null) {
  if (!provider.trim() || !model.trim()) throw new Error('Choose a provider and model first.')
  validateChatSettings(settings)
  const { state } = initialFormState(defaults, { provider, model, type: 'one_shot' })
  state.patient.systemPrompt = settings.system_prompt ? { mode: 'custom', text: settings.system_prompt } : { mode: 'none' }
  state.patient.generation = {
    temperature: settings.temperature == null ? '' : String(settings.temperature),
    top_p: settings.top_p == null ? '' : String(settings.top_p),
    max_tokens: settings.max_tokens == null ? '' : String(settings.max_tokens),
    enable_cot: settings.enable_cot, use_dynamic_strategies: false,
  }
  state.seed = settings.seed == null ? '' : String(settings.seed)
  state.oneShotSource = 'custom'
  state.customPrompt = prompt
  const request = buildTestRunRequest(state)
  const config = request.test_config || {}
  // The form normalizes authored text; a chat handoff retains the exact instructions.
  if (settings.system_prompt) config.patient_system_prompt = settings.system_prompt
  const query = { provider, model, type: 'one_shot', doctor_provider: state.doctor.provider, doctor_model: state.doctor.model, test_config: JSON.stringify(config) }
  const options = {
    system_prompt: settings.system_prompt, enable_cot: settings.enable_cot,
    ...(settings.temperature == null ? {} : { temperature: settings.temperature }),
    ...(settings.top_p == null ? {} : { top_p: settings.top_p }),
    ...(settings.max_tokens == null ? {} : { max_new_tokens: settings.max_tokens }),
    ...(settings.seed == null ? {} : { seed: settings.seed }),
  }
  return {
    test: { pathname: '/create-test', query },
    suite: { pathname: '/suite', query },
    modify: supportsWeightActions(provider) ? { pathname: '/weights', query: { source_model: model } } : null,
    compare: supportsWeightActions(provider) ? { pathname: '/weights', query: { kind: 'compare', source_model: sourceModel || model, ...(sourceModel ? { modified_model: model } : {}), options: JSON.stringify(options) } } : null,
  }
}

export function validateChatSettings(settings: ChatSettings) {
  if (settings.temperature != null && (!Number.isFinite(settings.temperature) || settings.temperature < 0 || settings.temperature > 2)) throw new Error('Temperature must be between 0 and 2.')
  if (settings.top_p != null && (!Number.isFinite(settings.top_p) || settings.top_p <= 0 || settings.top_p > 1)) throw new Error('Top-p must be greater than 0 and at most 1.')
  if (settings.max_tokens != null && (!Number.isInteger(settings.max_tokens) || settings.max_tokens < 1 || settings.max_tokens > 32768)) throw new Error('Maximum output tokens must be between 1 and 32768.')
  if (settings.seed != null && (!Number.isInteger(settings.seed) || settings.seed < 0 || settings.seed > 4294967295)) throw new Error('Seed must be a whole number between 0 and 4294967295.')
  if (characterCount(settings.system_prompt) > 32000) throw new Error('The system prompt exceeds 32,000 characters.')
}

/** Only completed exchanges enter the next request; retries cannot duplicate turns. */
export function buildChatRequest(exchanges: ChatExchange[], prompt: string, settings: ChatSettings) {
  if (!prompt.trim()) throw new Error('Enter a message before sending.')
  validateChatSettings(settings)
  // Empty final answers cannot be valid assistant history; reasoning is never a substitute.
  const messages = exchanges.filter((turn) => turn.response.content.trim()).flatMap((turn) => [
    { role: 'user' as const, content: turn.prompt },
    { role: 'assistant' as const, content: turn.response.content },
  ])
  messages.push({ role: 'user', content: prompt })
  if (messages.length > 100) throw new Error('This conversation has reached its 50-question limit. Export it and start a new conversation.')
  if (messages.some((message) => characterCount(message.content) > 32000)) throw new Error('A message exceeds 32,000 characters. Shorten it or start a new conversation.')
  if (messages.reduce((total, message) => total + characterCount(message.content), characterCount(settings.system_prompt)) > 128000) throw new Error('Conversation history and system prompt exceed 128,000 characters. Export it and start a new conversation.')
  return { ...settings, messages, system_prompt: settings.system_prompt || undefined }
}

export function chatErrorMessage(error: unknown): string {
  const candidate = error as { response?: { data?: { detail?: unknown } }; message?: string }
  const detail = candidate?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((issue) => typeof issue?.msg === 'string' ? issue.msg : 'Invalid request').join('; ')
  return candidate?.message || 'The model did not return a response. Your message is still here; you can retry.'
}
