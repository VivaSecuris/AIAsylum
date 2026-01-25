import { useState } from 'react'
import { Plus, X } from 'lucide-react'

interface PatientModel {
  id: string
  provider: string
  model: string
}

interface MultiPatientSelectorProps {
  patients: PatientModel[]
  onChange: (patients: PatientModel[]) => void
}

const PROVIDERS = ['openai', 'anthropic', 'google', 'ollama']

const MODELS_BY_PROVIDER: Record<string, string[]> = {
  openai: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'],
  anthropic: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-2', 'claude-instant'],
  google: ['gemini-pro', 'gemini-pro-vision', 'palm-2'],
  ollama: ['llama2', 'llama3', 'llama3.2', 'mistral', 'mixtral', 'codellama'],
}

export function MultiPatientSelector({ patients, onChange }: MultiPatientSelectorProps) {
  const addPatient = () => {
    const newPatient: PatientModel = {
      id: `patient_${Date.now()}`,
      provider: '',
      model: '',
    }
    onChange([...patients, newPatient])
  }

  const removePatient = (id: string) => {
    onChange(patients.filter((p) => p.id !== id))
  }

  const updatePatient = (id: string, field: 'provider' | 'model', value: string) => {
    onChange(
      patients.map((p) => {
        if (p.id === id) {
          const updated = { ...p, [field]: value }
          // Reset model if provider changes and model is not available for new provider
          if (field === 'provider' && p.model && !MODELS_BY_PROVIDER[value]?.includes(p.model)) {
            updated.model = ''
          }
          return updated
        }
        return p
      })
    )
  }

  const getAvailableModels = (provider: string): string[] => {
    return MODELS_BY_PROVIDER[provider] || []
  }

  return (
    <div className="space-y-3">
      {patients.map((patient, index) => (
        <div key={patient.id} className="rounded-lg border bg-card p-4">
          <div className="flex items-center justify-between mb-3">
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
                className="mt-1 w-full rounded border px-3 py-2 text-sm"
              >
                <option value="">Select provider</option>
                {PROVIDERS.map((p) => (
                  <option key={p} value={p}>
                    {p.charAt(0).toUpperCase() + p.slice(1)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground">Model</label>
              <select
                value={patient.model}
                onChange={(e) => updatePatient(patient.id, 'model', e.target.value)}
                disabled={!patient.provider}
                className="mt-1 w-full rounded border px-3 py-2 text-sm disabled:bg-muted"
              >
                <option value="">Select model</option>
                {getAvailableModels(patient.provider).map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      ))}
      <button
        type="button"
        onClick={addPatient}
        className="flex items-center gap-2 rounded-lg border border-dashed px-4 py-2 text-sm font-medium hover:bg-muted"
      >
        <Plus className="h-4 w-4" />
        Add Patient Model
      </button>
      {patients.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Click "Add Patient Model" to add at least one patient model to the group therapy session.
        </p>
      )}
    </div>
  )
}
