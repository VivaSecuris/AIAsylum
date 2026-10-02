import type { PreflightResponse, WeightRun, WeightRunRequest } from '@/lib/api'
import type { ChatExchange, ChatResponse, ChatSettings } from '@/lib/model-chat'
import { supportsWeightActions, validateChatSettings } from '@/lib/model-chat'

export type AdjustmentMode = 'profile' | 'weights'
export type AdjustmentMessage = { role: 'user' | 'assistant'; content: string }
export interface AdjustmentCoach {
  provider: string
  model: string
  temperature?: number | null
  top_p?: number | null
  max_tokens?: number | null
  seed?: number | null
  enable_cot?: boolean
}
export interface AdjustmentTarget {
  provider: string
  model: string
  settings: ChatSettings
}
export interface ModelAdjustmentRequest {
  provider: string
  model: string
  instruction: string
  mode: AdjustmentMode
  messages: AdjustmentMessage[]
  settings: ChatSettings
  coach: AdjustmentCoach
  baseline_evidence?: ChatResponse
  parent_id?: string
}
export interface ModelAdjustment {
  id: string
  provider: string
  model: string
  mode: AdjustmentMode
  status: 'proposed' | 'applying' | 'applied' | 'training' | 'failed'
  instruction: string
  settings: ChatSettings
  baseline: { messages: AdjustmentMessage[]; response?: ChatResponse | string; evidence?: ChatResponse; evidence_source?: 'client_reported' }
  proposal: {
    summary: string
    system_prompt: string
    training_examples: Array<{ prompt: string; response: string }>
    test_prompts: string[]
  }
  weight_request?: WeightRunRequest | null
  weight_run?: WeightRun | null
  active_target?: AdjustmentTarget | null
  tests: Array<{ at?: string; response: ChatResponse; target?: AdjustmentTarget }>
  coach?: AdjustmentCoach
  coach_evidence?: ChatResponse
  error?: string | null
  parent_id?: string | null
  created_at?: string
}

export function adjustmentMessages(exchanges: ChatExchange[]): AdjustmentMessage[] {
  return exchanges.filter((turn) => turn.response.content.trim()).flatMap((turn) => [
    { role: 'user' as const, content: turn.prompt },
    { role: 'assistant' as const, content: turn.response.content },
  ])
}

export function buildAdjustmentRequest(input: Omit<ModelAdjustmentRequest, 'messages'> & { exchanges: ChatExchange[] }): ModelAdjustmentRequest {
  validateChatSettings(input.settings)
  if (!input.instruction.trim()) throw new Error('Describe the behavior you want to change.')
  if (Array.from(input.instruction).length > 8000) throw new Error('Keep the adjustment instruction within 8,000 characters.')
  if (!input.coach.provider.trim() || !input.coach.model.trim()) throw new Error('Choose a coach model to propose the change.')
  if (input.mode === 'weights' && !supportsWeightActions(input.provider)) throw new Error('Training needs a local Transformers checkpoint.')
  const messages = adjustmentMessages(input.exchanges)
  if (!messages.length || !input.exchanges.at(-1)?.response.content.trim()) throw new Error('Get a complete model answer before requesting an adjustment.')
  if (messages.length > 100 || messages.reduce((total, message) => total + Array.from(message.content).length, Array.from(input.settings.system_prompt).length) > 80000) throw new Error('Use a shorter conversation for the adjustment.')
  const { exchanges: _, ...request } = input
  return { ...request, messages, baseline_evidence: input.exchanges.at(-1)!.response }
}

export function adjustmentPreflightParams(request: WeightRunRequest) {
  return {
    kind: request.kind, source_model: request.source_model, dtype: request.dtype,
    output_name: request.output_name, merge: request.merge,
  }
}

export function adjustmentCanApply(adjustment: ModelAdjustment, preflight: PreflightResponse | null, acknowledged: string[]) {
  if (adjustment.status !== 'proposed') return false
  if (adjustment.mode === 'profile') return true
  if (!adjustment.weight_request || !preflight) return false
  const acknowledgedCheck = (code: string) => preflight.checks.some((check) => check.code === code && check.acknowledgeable && acknowledged.includes(code))
  return (preflight.can_proceed || preflight.blocking_codes.length > 0) && preflight.blocking_codes.every(acknowledgedCheck) && preflight.checks.every((check) =>
    check.severity !== 'blocking' || acknowledgedCheck(check.code))
}

export function adjustmentCanUse(adjustment: ModelAdjustment) {
  return adjustment.status === 'applied' && !!adjustment.active_target
}
