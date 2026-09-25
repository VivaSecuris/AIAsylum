import type { apiClient } from '@/lib/api'

export type ChatResponse = Awaited<ReturnType<typeof apiClient.chatWithModel>>
export interface ChatSettings {
  system_prompt: string
  temperature: number
  max_tokens: number
  device: 'auto' | 'cpu' | 'cuda' | 'mps'
  dtype: 'float32' | 'float16' | 'bfloat16'
}
export interface ChatExchange {
  prompt: string
  response: ChatResponse
  at: string
}

const characterCount = (text: string) => Array.from(text).length

/** Only completed exchanges enter the next request; retries cannot duplicate turns. */
export function buildChatRequest(exchanges: ChatExchange[], prompt: string, settings: ChatSettings) {
  if (!prompt.trim()) throw new Error('Enter a message before sending.')
  if (!Number.isFinite(settings.temperature) || settings.temperature < 0 || settings.temperature > 2) throw new Error('Temperature must be between 0 and 2.')
  if (!Number.isInteger(settings.max_tokens) || settings.max_tokens < 1 || settings.max_tokens > 4096) throw new Error('Maximum output tokens must be between 1 and 4096.')
  if (characterCount(settings.system_prompt) > 32000) throw new Error('The system prompt exceeds 32,000 characters.')
  const messages = exchanges.flatMap((turn) => [
    { role: 'user' as const, content: turn.prompt },
    { role: 'assistant' as const, content: turn.response.content },
  ])
  messages.push({ role: 'user', content: prompt })
  if (messages.length > 100) throw new Error('This conversation has reached its 50-question limit. Export it and start a new conversation.')
  if (messages.some((message) => characterCount(message.content) > 32000)) throw new Error('A message exceeds 32,000 characters. Shorten it or start a new conversation.')
  if (messages.reduce((total, message) => total + characterCount(message.content), 0) > 128000) throw new Error('Conversation history exceeds 128,000 characters. Export it and start a new conversation.')
  return { ...settings, messages, system_prompt: settings.system_prompt || undefined }
}

export function chatErrorMessage(error: unknown): string {
  const candidate = error as { response?: { data?: { detail?: unknown } }; message?: string }
  const detail = candidate?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((issue) => typeof issue?.msg === 'string' ? issue.msg : 'Invalid request').join('; ')
  return candidate?.message || 'The model did not return a response. Your message is still here; you can retry.'
}
