import type { WeightRunKind, WeightRunRequest } from './api'

/** Visible state for the newer weight experiments, shared by restore and submit. */
export interface NewKindState {
  probeRunId: string
  hneuronsRunId: string
  categoryName: string
  categoryPrompts: string
  categoryNearMiss: string
  ms: string
  taus: string
  nQuestions: number
  nSamples: number
  hneuronAlpha: number
  redteamTarget: 'refusal' | 'prompt_leak'
  secretSystem: string
  baselineModel: string
  modelB: string
  referenceModel: string
  whichEmbedding: 'input' | 'output'
  nQueries: number
  colSubset: number
}

export const NEW_KIND_DEFAULTS: NewKindState = {
  probeRunId: '', hneuronsRunId: '', categoryName: '', categoryPrompts: '', categoryNearMiss: '',
  ms: '0.5,0.75,1.0,1.25,1.5', taus: '0.5,0.6,0.7,0.8,0.9',
  nQuestions: 400, nSamples: 10, hneuronAlpha: 0.5,
  redteamTarget: 'refusal', secretSystem: '', baselineModel: '',
  modelB: '', referenceModel: '', whichEmbedding: 'input', nQueries: 2048, colSubset: 4096,
}

export const NEW_KINDS: WeightRunKind[] = [
  'induce', 'hneurons', 'hneuron_bake', 'redteam', 'embed_align', 'embed_extract', 'embed_recon',
]

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const stringOr = (value: unknown, fallback: string) => typeof value === 'string' ? value : fallback
const numberOr = (value: unknown, fallback: number) => finite(value) ? value : fallback
const runId = (value: unknown) => (finite(value) && Number.isInteger(value) && value > 0)
  || (typeof value === 'string' && /^[1-9]\d*$/.test(value)) ? String(value) : ''
const lines = (value: unknown) => Array.isArray(value) ? value.filter((s) => typeof s === 'string').join('\n') : ''
const grid = (value: unknown, fallback: string) => Array.isArray(value) ? value.filter(finite).join(',') : fallback

/** A fresh state prevents values leaking between two lineage links in one page. */
export function restoreNewKindState(options: Record<string, unknown> = {}): NewKindState {
  const defaults = NEW_KIND_DEFAULTS
  const cfg = options.category_config && typeof options.category_config === 'object'
    ? options.category_config as Record<string, unknown> : {}
  return {
    probeRunId: runId(options.probe_run_id),
    hneuronsRunId: runId(options.hneurons_run_id),
    categoryName: stringOr(cfg.name, defaults.categoryName),
    categoryPrompts: lines(cfg.prompts),
    categoryNearMiss: lines(cfg.near_miss),
    ms: grid(options.ms, defaults.ms),
    taus: grid(options.taus, defaults.taus),
    nQuestions: numberOr(options.n_questions, defaults.nQuestions),
    nSamples: numberOr(options.n_samples, defaults.nSamples),
    hneuronAlpha: numberOr(options.hneuron_alpha, defaults.hneuronAlpha),
    redteamTarget: options.redteam_target === 'prompt_leak' ? 'prompt_leak' : 'refusal',
    secretSystem: stringOr(options.secret_system, defaults.secretSystem),
    baselineModel: stringOr(options.baseline_model, defaults.baselineModel),
    modelB: stringOr(options.model_b, defaults.modelB),
    referenceModel: stringOr(options.reference_model, defaults.referenceModel),
    whichEmbedding: options.which_embedding === 'output' ? 'output' : 'input',
    nQueries: numberOr(options.n_queries, defaults.nQueries),
    colSubset: numberOr(options.col_subset, defaults.colSubset),
  }
}

const splitLines = (value: string) => value.split('\n').map((s) => s.trim()).filter(Boolean)
const parseGrid = (value: string) => value.split(',').map((s) => Number.parseFloat(s.trim())).filter(Number.isFinite)

/** Rebuild every visible field so clearing a restored input also clears the request. */
export function newKindRequest(kind: WeightRunKind, state: NewKindState): Partial<WeightRunRequest> {
  // Other stages can use these options too (for example a category probe).
  // Their hidden recorded inputs are not controlled by this form.
  if (!NEW_KINDS.includes(kind)) return {}
  return {
    probe_run_id: kind === 'induce' && state.probeRunId ? Number(state.probeRunId) : undefined,
    category_config: kind === 'induce' || (kind === 'redteam' && state.redteamTarget === 'refusal')
      ? { name: state.categoryName || 'category', prompts: splitLines(state.categoryPrompts), near_miss: splitLines(state.categoryNearMiss) }
      : undefined,
    ms: kind === 'induce' ? parseGrid(state.ms) : undefined,
    taus: kind === 'induce' ? parseGrid(state.taus) : undefined,
    n_questions: kind === 'hneurons' ? state.nQuestions : undefined,
    n_samples: kind === 'hneurons' ? state.nSamples : undefined,
    hneurons_run_id: kind === 'hneuron_bake' && state.hneuronsRunId ? Number(state.hneuronsRunId) : undefined,
    hneuron_alpha: kind === 'hneuron_bake' ? state.hneuronAlpha : undefined,
    redteam_target: kind === 'redteam' ? state.redteamTarget : undefined,
    baseline_model: kind === 'redteam' && state.baselineModel.trim() ? state.baselineModel.trim() : undefined,
    secret_system: kind === 'redteam' && state.redteamTarget === 'prompt_leak' ? state.secretSystem : undefined,
    model_b: kind === 'embed_align' ? state.modelB.trim() : undefined,
    reference_model: kind === 'embed_recon' ? state.referenceModel.trim() : undefined,
    which_embedding: kind === 'embed_align' ? state.whichEmbedding : undefined,
    n_queries: kind === 'embed_extract' || kind === 'embed_recon' ? state.nQueries : undefined,
    col_subset: kind === 'embed_extract' ? state.colSubset : undefined,
  }
}
