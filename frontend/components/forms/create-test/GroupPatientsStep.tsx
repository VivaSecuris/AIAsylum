import { Plus, Scissors, X } from 'lucide-react'

import type { GroupPatient, StepState } from '@/lib/create-test-config'
import { newGroupPatient } from '@/lib/create-test-config'
import { useEditedModels } from '@/lib/hooks'
import { RoleFields } from './RoleStep'
import { SMALL_BUTTON } from './ui'

/** Group therapy: every patient is its own model with its own system prompt and sampling. */
export function GroupPatientsStep({ patients, onChange, defaults }: {
  defaults: StepState
  patients: GroupPatient[]
  onChange: (patients: GroupPatient[]) => void
}) {
  const { data: edited = [] } = useEditedModels()
  const replace = (key: string, next: GroupPatient) => onChange(patients.map((p) => (p.key === key ? next : p)))
  const add = (provider = defaults.provider, model = defaults.model) => onChange([...patients,
    newGroupPatient(provider, model, defaults.systemPrompt.mode === 'none' ? { mode: 'default' } : defaults.systemPrompt, defaults.generation)])

  return (
    <div className="space-y-4">
      {patients.map((patient, index) => (
        <div key={patient.key} className="space-y-3 rounded-md border p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Patient {index + 1}</h3>
            <button type="button" className={SMALL_BUTTON} disabled={patients.length <= 1}
              onClick={() => onChange(patients.filter((p) => p.key !== patient.key))} aria-label={`Remove patient ${index + 1}`}>
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <RoleFields role="patient" step={patient} onChange={(next) => replace(patient.key, { ...patient, ...next })} />
        </div>
      ))}
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => add()}
          className="flex items-center gap-2 rounded-lg border border-dashed px-4 py-2 text-sm font-medium hover:bg-muted">
          <Plus className="h-4 w-4" /> Add patient
        </button>
        {Array.isArray(edited) && edited.length > 0 && (
          <>
            <span className="text-xs text-muted-foreground">or add an edited model:</span>
            {edited.slice(0, 4).map((m: any) => (
              <button key={m.path} type="button" title={m.path} onClick={() => add('transformers', m.path)}
                className="flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm hover:bg-muted">
                <Scissors className="h-3.5 w-3.5" /> {m.name}
              </button>
            ))}
          </>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        Group therapy needs at least two patients. To measure an edit, add the edited model and the stock model it came from.
      </p>
    </div>
  )
}
