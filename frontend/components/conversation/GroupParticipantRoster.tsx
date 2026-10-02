import { ModelChatLink } from '@/components/models/ModelChatLink'
import { compactModelName, type GroupParticipant } from '@/lib/group-participants'
import { participantAccent } from './participant-colors'

export function GroupParticipantRoster({ participants }: { participants: GroupParticipant[] }) {
  if (!participants.length) return null
  return <section className="rounded-lg border bg-card p-4" aria-label="Group participants">
    <h3 className="mb-3 font-semibold">Patient Models ({participants.length})</h3>
    <p className="mb-3 text-xs text-muted-foreground">
      Round robin: Doctor → {participants.map(patient => `Patient ${patient.number}`).join(' → ')}.
      {' '}Each reply joins the shared conversation before the next turn.
    </p>
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {participants.map(patient => <div key={patient.id} className={`min-w-0 rounded-lg border-l-4 p-3 ${participantAccent(patient)}`}>
        <p className="text-xs font-semibold">Patient {patient.number}{patient.baseline && <span className="ml-2 rounded border border-current/30 px-1.5 py-0.5">Baseline</span>}</p>
        <p className="mt-1 break-words text-sm font-semibold"><ModelChatLink provider={patient.provider} model={patient.model}>{patient.name}</ModelChatLink></p>
        <p className="mt-1 text-xs text-muted-foreground">{patient.provider}{patient.method && ` · ${patient.method}`}</p>
        {patient.baseline && <p className="mt-1 text-xs">Source checkpoint for the modified models</p>}
        {patient.sourceModel && <p className="mt-1 break-words text-xs" title={patient.sourceModel}>From {compactModelName(patient.sourceModel)}</p>}
        <details className="mt-2 text-xs text-muted-foreground">
          <summary className="cursor-pointer">Exact model reference</summary>
          <p className="mt-1 break-all">{patient.model}</p>
        </details>
      </div>)}
    </div>
  </section>
}
