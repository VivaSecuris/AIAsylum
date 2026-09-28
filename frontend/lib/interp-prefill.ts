import type { InterpMode } from './api'

const FLAG_NAMES = ['enable_attention_capture', 'enable_mlp_capture', 'enable_qkv_capture', 'enable_pre_mlp_capture', 'enable_patching', 'enable_scrub', 'enable_minimal_circuit']

/** Decode a fresh form snapshot; never merge it with a previous branch's state. */
export function parseInterpPrefill(query: Record<string, string | string[] | undefined>, defaultModel = 'Qwen/Qwen2.5-0.5B-Instruct') {
  let options: Record<string, unknown> = {}
  const warnings: string[] = []
  if (typeof query.options === 'string') {
    try {
      const parsed = JSON.parse(query.options)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Invalid options')
      options = parsed
    } catch { warnings.push('The saved options could not be read. Review the default settings before running.') }
  }
  const text = (key: string, fallback = '') => typeof query[key] === 'string' ? query[key] as string : typeof options[key] === 'string' ? options[key] as string : fallback
  const integer = (key: string, fallback: number) => {
    if (typeof options[key] === 'number' && Number.isInteger(options[key])) return options[key] as number
    if (options[key] != null) warnings.push(`The saved ${key} value could not be restored.`)
    return fallback
  }
  const modeValue = text('mode', 'single')
  const mode = (['single', 'comparison', 'progression', 'model_diff'].includes(modeValue) ? modeValue : 'single') as InterpMode
  let prompts: string[] = []
  let rawPrompts: unknown = options.prompts
  if (typeof query.prompts === 'string') {
    try { rawPrompts = JSON.parse(query.prompts) } catch { warnings.push('The saved prompt list could not be read. Add the prompts again before running.') }
  }
  if (Array.isArray(rawPrompts) && rawPrompts.every((item) => typeof item === 'string')) prompts = rawPrompts
  const deviceValue = text('device', 'auto')
  const dtypeValue = text('dtype', 'bfloat16')
  const precision: Record<string, string> = { bf16: 'bfloat16', fp16: 'float16', fp32: 'float32' }
  const device = /^(auto|cpu|mps|cuda(?::\d+)?)$/.test(deviceValue) ? deviceValue : 'auto'
  const dtype = precision[dtypeValue] ?? (['bfloat16', 'float16', 'float32'].includes(dtypeValue) ? dtypeValue : 'bfloat16')
  if (device !== deviceValue) warnings.push('The saved device is unsupported. Automatic device selection is shown.')
  if (!['bf16', 'fp16', 'fp32', 'bfloat16', 'float16', 'float32'].includes(dtypeValue)) warnings.push('The saved precision is unsupported. BF16 is shown.')
  const patchComponent = ['layer', 'head', 'neuron'].includes(String(options.patch_components ?? 'layer')) ? String(options.patch_components ?? 'layer') as 'layer' | 'head' | 'neuron' : 'layer'
  if (options.patch_components != null && options.patch_components !== patchComponent) warnings.push('The saved patch component is unsupported. Residual patching is shown; review the intervention before running.')
  const patchList = (key: string, pairs = false): string => {
    const value = options[key]
    if (value == null) return ''
    const index = (item: unknown) => typeof item === 'number' && Number.isInteger(item) && item >= 0
    if (!Array.isArray(value) || value.length < 1 || value.length > 64 || !value.every((item) => pairs ? Array.isArray(item) && item.length === 2 && item.every(index) : index(item))) {
      warnings.push(`The saved ${key} selection could not be restored. Review the patch settings before running.`)
      return ''
    }
    return value.map((item) => pairs ? item.join(':') : String(item)).join(', ')
  }
  const patch = { component: patchComponent, layers: patchList('patch_layers'), positions: patchList('patch_positions'), units: patchComponent === 'head' ? patchList('patch_heads', true) : patchComponent === 'neuron' ? patchList('patch_neurons', true) : '' }
  return {
    mode, patch,
    modelA: text('model_a', defaultModel), modelB: text('model_b'),
    promptA: typeof query.prompt_a === 'string' ? query.prompt_a : text('prompt', typeof options.prompt_a === 'string' ? options.prompt_a : ''),
    promptB: text('prompt_b'), prompts, device, dtype,
    maxLen: integer('max_len', 512), window: integer('window', 128), topk: integer('topk', 10),
    dimReduction: ['pca', 'tsne', 'umap'].includes(text('dim_reduction', 'pca')) ? text('dim_reduction', 'pca') : 'pca',
    advanced: Object.fromEntries(FLAG_NAMES.map((name) => [name, options[name] === true])), warnings,
  }
}
