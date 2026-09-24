import { useEffect, useMemo, useState } from 'react'
import { Plus, Scissors, X } from 'lucide-react'

import { apiClient } from '@/lib/api'
import type { EditedModel, ProviderInfo } from '@/lib/api'

interface PatientModel {
  id: string
  provider: string
  model: string
}

interface MultiPatientSelectorProps {
  patients: PatientModel[]
  onChange: (patients: PatientModel[]) => void
}

const CONTROL =
  'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50'

/**
 * Pick the models a suite runs against.
 *
 * The provider list comes from the backend rather than a constant here. It used
 * to be hardcoded to four names, which silently excluded `transformers` -- and
 * so made surgically edited models impossible to test beside a stock one, which
 * is the entire point of editing them. That is the same failure the providers
 * endpoint exists to prevent: registered in the backend, unreachable from the app.
 */
export function MultiPatientSelector({ patients, onChange }: MultiPatientSelectorProps) {
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [ollamaModels, setOllamaModels] = useState<string[]>([])
  const [ollamaLoading, setOllamaLoading] = useState(false)
  const [edited, setEdited] = useState<EditedModel[]>([])

  useEffect(() => {
    let cancelled = false
    apiClient.listProviders()
      .then((list) => { if (!cancelled) setProviders(Array.isArray(list) ? list : []) })
      .catch(() => { if (!cancelled) setProviders([]) })
    // Edited models are a convenience shortcut; a failure here is not an error,
    // it just means the button does not appear.
    apiClient.listEditedModels()
      .then((list) => { if (!cancelled) setEdited(Array.isArray(list) ? list : []) })
      .catch(() => { if (!cancelled) setEdited([]) })
    return () => { cancelled = true }
  }, [])

  const hasOllama = patients.some((p) => p.provider === 'ollama')
  useEffect(() => {
    if (!hasOllama) return
    let cancelled = false
    setOllamaLoading(true)
    apiClient.listOllamaModels()
      .then((list) => { if (!cancelled) setOllamaModels(Array.isArray(list) ? list : []) })
      .catch(() => { if (!cancelled) setOllamaModels([]) })
      .finally(() => { if (!cancelled) setOllamaLoading(false) })
    return () => { cancelled = true }
  }, [hasOllama])

  const byName = useMemo(
    () => Object.fromEntries(providers.map((p) => [p.name, p])),
    [providers],
  )

  const addPatient = (provider = '', model = '') => {
    onChange([
      ...patients,
      { id: `patient_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`, provider, model },
    ])
  }

  const removePatient = (id: string) => onChange(patients.filter((p) => p.id !== id))

  const updatePatient = (id: string, field: 'provider' | 'model', value: string) => {
    onChange(
      patients.map((p) => {
        if (p.id !== id) return p
        const updated = { ...p, [field]: value }
        // Only clear the model when switching to a provider that offers a fixed
        // list and does not contain it. A typed path must survive.
        if (field === 'provider' && p.model) {
          const info = byName[value]
          if (info?.model_input === 'list' && !(info.models || []).includes(p.model)) {
            updated.model = ''
          } else if (info?.model_input === 'fetch' && !ollamaModels.includes(p.model)) {
            updated.model = ''
          }
        }
        return updated
      }),
    )
  }

  function ModelField({ patient }: { patient: PatientModel }) {
    const info = byName[patient.provider]

    if (!info || info.model_input === 'path' || info.model_input === 'text') {
      return (
        <input
          type="text"
          value={patient.model}
          disabled={!patient.provider}
          placeholder={info?.placeholder || 'Model id or local path'}
          onChange={(e) => updatePatient(patient.id, 'model', e.target.value)}
          className={CONTROL}
        />
      )
    }

    const options = info.model_input === 'fetch' ? ollamaModels : info.models || []
    return (
      <select
        value={patient.model}
        onChange={(e) => updatePatient(patient.id, 'model', e.target.value)}
        disabled={!patient.provider || (info.model_input === 'fetch' && ollamaLoading)}
        className={CONTROL}
      >
        <option value="">
          {info.model_input === 'fetch' && ollamaLoading ? 'Loading…' : 'Select model'}
        </option>
        {options.map((m) => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
    )
  }

  return (
    <div className="space-y-3">
      {patients.map((patient, index) => {
        const info = byName[patient.provider]
        return (
          <div key={patient.id} className="rounded-lg border bg-card p-4">
            <div className="mb-3 flex items-center justify-between">
              <label className="text-sm font-medium">Patient {index + 1}</label>
              {patients.length > 1 && (
                <button
                  type="button"
                  onClick={() => removePatient(patient.id)}
                  className="text-red-600 hover:text-red-800"
                  title="Remove patient"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted-foreground">Provider</label>
                <select
                  value={patient.provider}
                  onChange={(e) => updatePatient(patient.id, 'provider', e.target.value)}
                  className={CONTROL}
                >
                  <option value="">Select provider</option>
                  {providers.map((p) => (
                    <option key={p.name} value={p.name} disabled={!p.available}>
                      {p.label}
                      {!p.available ? ' — unavailable' : ''}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Model</label>
                <ModelField patient={patient} />
              </div>
            </div>
            {info && !info.configured && info.requires_api_key && (
              <p className="mt-2 text-xs text-muted-foreground">
                {info.label} has no API key configured, so this patient will fail at run time.
              </p>
            )}
            {info?.unavailable_reason && (
              <p className="mt-2 text-xs text-destructive">{info.unavailable_reason}</p>
            )}
          </div>
        )
      })}

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => addPatient()}
          className="flex items-center gap-2 rounded-lg border border-dashed px-4 py-2 text-sm font-medium hover:bg-muted"
        >
          <Plus className="h-4 w-4" />
          Add Patient Model
        </button>

        {/* The point of a surgically edited model is comparing it against the
            stock one it came from, in the same suite. */}
        {edited.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">or add an edited model:</span>
            {edited.slice(0, 4).map((m) => (
              <button
                key={m.path}
                type="button"
                onClick={() => addPatient('transformers', m.path)}
                title={`${m.path}${m.manifest?.beta != null ? ` (β=${m.manifest.beta})` : ''}`}
                className="flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm hover:bg-muted"
              >
                <Scissors className="h-3.5 w-3.5" />
                {m.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {patients.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Add at least one patient model. To measure an edit, add the edited model and the
          stock model it came from — run through the same provider, the weights are then the
          only variable.
        </p>
      )}
    </div>
  )
}
