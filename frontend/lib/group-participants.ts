import type { ConversationTurn } from './api'
import type { ModelCatalogEntry } from './model-catalog'

export interface GroupPatient {
  id?: number
  provider: string
  model: string
}

const METHOD_NAMES: Record<string, string> = {
  lora_merge: 'LoRA merged',
  direction_subspace: 'Direction subspace edit',
}

export interface GroupParticipant extends GroupPatient {
  id: number
  number: number
  name: string
  baseline: boolean
  sourceModel?: string
  method?: string
}

/** Shorten filesystem paths, preserving Hub/provider references and exact link targets. */
export function compactModelName(model: string): string {
  return /^(\/|[A-Za-z]:[\\/])/.test(model)
    ? model.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || model
    : model
}

export function buildGroupParticipants(
  patients: GroupPatient[] = [],
  turns: ConversationTurn[] = [],
  catalog: ModelCatalogEntry[] = [],
): GroupParticipant[] {
  const entryFor = (provider: string, model: string) => catalog.find(entry =>
    entry.provider === provider && (entry.model_ref === model || entry.aliases?.includes(model)))
  const canonical = (provider: string, model: string) => entryFor(provider, model)?.model_ref || model
  const identity = (provider: string, model: string) => `${provider}\0${canonical(provider, model)}`
  const sources = new Map<string, string>()
  const methods = new Map<string, string>()
  const knownBase = new Set<string>()
  for (const entry of catalog) {
    const key = identity(entry.provider, entry.model_ref)
    if (entry.source_model) sources.set(key, entry.source_model)
    if (entry.manifest?.method) methods.set(key, String(entry.manifest.method))
    if (entry.kind === 'base' && !entry.source_model) knownBase.add(key)
  }
  // Generation-time evidence takes precedence over today's catalog for historical runs.
  for (const turn of turns) {
    if (turn.speaker !== 'patient' || !turn.model_provider || !turn.model_name) continue
    const generation = turn.metadata?.generation_metadata
    if (!generation || !Object.prototype.hasOwnProperty.call(generation, 'surgery')) continue
    const key = identity(turn.model_provider, turn.model_name)
    const surgery = generation.surgery
    sources.delete(key)
    methods.delete(key)
    knownBase.delete(key)
    if (surgery === null) knownBase.add(key)
    else if (surgery && typeof surgery === 'object') {
      if (typeof surgery.source_model === 'string') sources.set(key, surgery.source_model)
      if (typeof surgery.method === 'string') methods.set(key, surgery.method)
    }
  }
  const ancestors = new Set<string>()
  for (const patient of patients) {
    let key = identity(patient.provider, patient.model)
    const visited = new Set<string>([key])
    while (sources.has(key)) {
      key = identity(patient.provider, sources.get(key)!)
      if (visited.has(key)) break
      visited.add(key)
      ancestors.add(key)
    }
  }
  return patients.map((patient, index) => {
    const key = identity(patient.provider, patient.model)
    return {
      ...patient,
      id: patient.id ?? index,
      number: index + 1,
      name: compactModelName(patient.model),
      baseline: knownBase.has(key) && ancestors.has(key) && !sources.has(key),
      sourceModel: sources.get(key),
      method: methods.has(key) ? METHOD_NAMES[methods.get(key)!] || methods.get(key)!.replace(/_/g, ' ') : undefined,
    }
  })
}

/** Never replace a recorded model with a requested participant that does not match it. */
export function participantForTurn(turn: ConversationTurn, participants: GroupParticipant[]): GroupParticipant | undefined {
  if (turn.speaker !== 'patient') return undefined
  const model = turn.model_name || turn.metadata?.patient_model
  if (!model) return undefined
  const matches = participants.filter(patient => patient.model === model
    && (!turn.model_provider || patient.provider === turn.model_provider))
  const id = turn.metadata?.patient_id
  return id !== undefined ? matches.find(patient => patient.id === id) : matches.length === 1 ? matches[0] : undefined
}

export function participantTitle(participant: GroupParticipant): string {
  return `Patient ${participant.number}${participant.baseline ? ' · Baseline' : ''} · ${participant.name}`
}
