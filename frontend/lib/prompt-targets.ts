import type { Prompt } from './api'

/** Patient test inputs may use general legacy prompts, but never another role's instructions. */
export function isPatientTestPrompt(prompt: Pick<Prompt, 'prompt_type' | 'target'>): boolean {
  return prompt.prompt_type === 'test_prompt' && (!prompt.target || prompt.target === 'patient')
}
