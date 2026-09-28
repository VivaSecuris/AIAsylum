import type { ProviderInfo, TestRun } from './api'
import type { ModelCatalogEntry } from './model-catalog'
import type { AppSettings } from './settings'

export interface ModelIdentity { provider: string; model: string }
export interface OverviewModel extends ModelIdentity {
  key: string
  name: string
  providerLabel: string
  kind: 'checkpoint' | 'service' | 'api'
  status: string
  ready: boolean
  custom: boolean
  reason?: string | null
  catalog?: ModelCatalogEntry
  runs: number
}

export const modelIdentityKey = ({ provider, model }: ModelIdentity) => JSON.stringify([provider, model])

/** Model identity includes the provider: identical names can serve different weights. */
export function buildModelOverview(
  catalog: ModelCatalogEntry[], providers: ProviderInfo[], ollamaModels: string[] | undefined,
  runs: TestRun[] = [], defaults?: AppSettings,
): OverviewModel[] {
  const models = new Map<string, OverviewModel>()
  function add(provider: string, model: string, entry?: ModelCatalogEntry) {
    if (!provider || !model) return undefined
    const info = providers.find(item => item.name === provider || item.aliases.includes(provider))
    const canonical = info?.name || (provider === 'local' ? 'transformers' : provider)
    entry = entry || catalog.find(item => (item.model_ref === model || item.aliases?.includes(model)) && (item.provider === canonical || canonical === 'transformers' && item.provider === 'local'))
    if (entry?.availability === 'deleted') return undefined
    model = entry?.model_ref || model
    const key = modelIdentityKey({ provider: canonical, model })
    if (models.has(key)) return models.get(key)
    const custom = entry?.kind === 'custom'
    const kind = canonical === 'transformers' ? 'checkpoint' : info?.kind === 'api' ? 'api' : 'service'
    let ready = false, status = 'Check availability', reason = entry?.reason
    if (info && !info.available) { status = 'Provider unavailable'; reason = info.unavailable_reason }
    else if (info?.requires_api_key && !info.configured) { status = 'Set up provider'; reason = 'Configure the provider in Settings.' }
    else if (entry) {
      ready = entry.availability === 'ready'
      status = ({ ready: 'Ready on server', download_required: 'Download required', missing: 'Checkpoint missing', incomplete: 'Incomplete checkpoint', deleted: 'Deleted' })[entry.availability]
    } else if (canonical === 'ollama') {
      ready = ollamaModels?.includes(model) ?? false
      status = ollamaModels === undefined ? 'Availability not verified' : ready ? 'Installed in Ollama' : 'Not installed in Ollama'
    } else if (kind === 'checkpoint') {
      status = 'Check checkpoint availability'
    } else if (info?.available && (!info.requires_api_key || info.configured)) {
      ready = true; status = 'Provider configured'
    }
    const item: OverviewModel = { key, provider: canonical, model, name: entry?.name || model.split('/').filter(Boolean).pop() || model, providerLabel: info?.label || canonical, kind, custom, ready, status, reason, catalog: entry, runs: entry?.history?.test_runs || 0 }
    models.set(key, item)
    return item
  }
  catalog.forEach(item => add(item.provider, item.model_ref, item))
  ollamaModels?.forEach(model => add('ollama', model))
  providers.forEach(provider => provider.models.forEach(model => add(provider.name, model)))
  const seenRuns = new Map<string, Set<number>>()
  runs.forEach(run => {
    const identities = [{ provider: run.doctor_provider, model: run.doctor_model }]
    const patients = run.meta_data?.test_config?.patients || run.meta_data?.patients
    if (run.test_type === 'group_therapy' && Array.isArray(patients) && patients.length) identities.push(...patients)
    else identities.push({ provider: run.patient_provider, model: run.patient_model })
    identities.forEach(identity => {
      const item = add(identity.provider, identity.model)
      if (!item) return
      const ids = seenRuns.get(item.key) || new Set<number>()
      ids.add(run.id); seenRuns.set(item.key, ids)
      item.runs = Math.max(item.runs, ids.size)
    })
  })
  if (defaults) {
    add(defaults.defaultDoctorProvider, defaults.defaultDoctorModel)
    add(defaults.defaultPatientProvider, defaults.defaultPatientModel)
    add(defaults.defaultEvaluatorProvider, defaults.defaultEvaluatorModel)
  }
  return Array.from(models.values()).sort((a, b) => Number(b.ready) - Number(a.ready) || Number(b.custom) - Number(a.custom) || a.name.localeCompare(b.name, undefined, { numeric: true }) || a.provider.localeCompare(b.provider))
}

/** Comparison handoffs are selections only; opening a URL never launches a run. */
export function parseModelSelection(value: unknown): ModelIdentity[] {
  if (typeof value !== 'string' || value.length > 50000) return []
  try {
    const raw: unknown = JSON.parse(value)
    if (!Array.isArray(raw) || raw.length > 20) return []
    const unique = new Map<string, ModelIdentity>()
    for (const item of raw) {
      if (!item || typeof item.provider !== 'string' || !item.provider.trim() || typeof item.model !== 'string' || !item.model.trim() || item.model.length > 2048) return []
      const identity = { provider: item.provider.trim(), model: item.model.trim() }
      unique.set(modelIdentityKey(identity), identity)
    }
    return Array.from(unique.values())
  } catch { return [] }
}

export function modelComparisonUrl(models: ModelIdentity[]) {
  return { pathname: '/suite', query: { type: 'one_shot', models: JSON.stringify(models.map(({ provider, model }) => ({ provider, model }))) } }
}
