import { ConversationTurn } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import { cn } from '@/lib/utils'
import { ModelChatLink } from '@/components/models/ModelChatLink'

/** Label recorded reasoning by its source, without claims about hidden internals. */
export function reasoningLabelFor(source?: string): string {
  if (source === 'react') return 'Requested reasoning (ReACT)'
  if (source === 'inline' || source === 'provider') return 'Model-provided reasoning'
  return 'Recorded reasoning (source not recorded)'
}

export function TurnFinishStatus({ reason }: { reason?: string | null }) {
  if (!reason) return null
  const limited = ['length', 'max_tokens', 'max_output_tokens'].includes(reason)
  return <p className="mt-2 text-xs"><span className="opacity-75">Finish: {reason}</span>{limited && <span className="ml-2 font-medium">Output reached its token limit; the response may be incomplete.</span>}</p>
}

/** Recorded response identity; legacy patient metadata never identifies a doctor. */
export function turnModelLabel(turn: ConversationTurn): string {
  const model = turn.model_name?.trim()
    || (turn.speaker === 'patient' ? turn.metadata?.patient_model?.trim() : '')
  return [turn.model_provider?.trim(), model].filter(Boolean).join(' / ')
}

/** Attribute input only when the preceding recorded response establishes its author. */
export function turnPromptLabel(turn: ConversationTurn, previous?: ConversationTurn): string {
  if (previous?.response?.trim() && previous.response.trim() === turn.prompt?.trim() && previous.speaker !== turn.speaker) {
    if (previous.speaker === 'doctor') return 'Replying to 👨‍⚕️ Doctor'
    if (previous.speaker === 'patient') return `Replying to 🤖 ${previous.metadata?.patient_name || 'Patient'}`
  }
  return 'Input prompt'
}

/** Exact system messages recorded from this turn's prepared generation request. */
export function TurnSystemPrompts({ prompts }: { prompts?: string[] }) {
  if (!Array.isArray(prompts)) return null

  return (
    <details className="mt-3 rounded border border-current/20 bg-black/5 p-2 text-xs">
      <summary className="cursor-pointer font-medium">System prompt sent</summary>
      {prompts.length === 0 ? (
        <p className="mt-2">No system message was supplied by the app. The provider or chat template may add its own default.</p>
      ) : prompts.map((prompt, index) => (
        <div key={index} className="mt-2">
          {prompts.length > 1 && <p className="mb-1 font-medium">System message {index + 1}</p>}
          <pre className="whitespace-pre-wrap break-words font-sans">{prompt}</pre>
        </div>
      ))}
    </details>
  )
}

interface ConversationViewerProps {
  turns: ConversationTurn[]
}

export function ConversationViewer({ turns }: ConversationViewerProps) {
  if (!turns || turns.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
        <p className="text-sm font-medium mb-2">No conversation turns available</p>
        <p className="text-xs">Conversation turns will appear here once the test run generates them.</p>
      </div>
    )
  }

  // Color scheme for different patients in group therapy
  const patientColors = [
    'bg-gray-50 border-gray-200',
    'bg-green-50 border-green-200',
    'bg-purple-50 border-purple-200',
    'bg-yellow-50 border-yellow-200',
    'bg-pink-50 border-pink-200',
    'bg-indigo-50 border-indigo-200',
  ]

  return (
    <div className="space-y-4">
      {turns.map((turn, index) => {
        const patientName = turn.metadata?.patient_name
        const patientId = turn.metadata?.patient_id
        const modelLabel = turnModelLabel(turn)

        const isGroupTherapyPatient = turn.speaker === 'patient' && patientName !== undefined
        const patientColorIndex = patientId !== undefined ? (patientId % patientColors.length) : 0
        const patientColorClass = isGroupTherapyPatient ? patientColors[patientColorIndex] : 'bg-gray-50 border-gray-200 dark:bg-gray-900 dark:border-gray-700'

        const isDoctor = turn.speaker === 'doctor'
        const nativeReasoning = turn.metadata?.native_reasoning || turn.metadata?.generation_metadata?.native_reasoning
        const reasoning = turn.metadata?.reasoning
        const reasoningLabel = reasoningLabelFor(turn.metadata?.reasoning_source)
        const hasPrompt = turn.prompt && turn.prompt.trim().length > 0
        const hasResponse = turn.response && turn.response.trim().length > 0

        const promptLabel = turnPromptLabel(turn, turns[index - 1])

        return (
          <div key={turn.id} className="space-y-1">
            {/* Compact context: what this AI is responding to (other speaker's message) */}
            {hasPrompt && (
              <div className="ml-4 text-xs text-muted-foreground italic border-l-2 border-muted pl-2">
                <span className="font-medium">{promptLabel}: </span>
                <span>{turn.prompt.length > 180 ? `${turn.prompt.substring(0, 180)}…` : turn.prompt}</span>
              </div>
            )}

            {/* Main message: this speaker's response (one bubble per turn, no duplicate) */}
            {(
              <div
                className={cn(
                  'rounded-lg border p-4',
                  isDoctor
                    ? 'bg-blue-50 border-blue-200 dark:bg-blue-950 dark:border-blue-800 ml-8'
                    : patientColorClass + ' mr-8'
                )}
              >
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="min-w-0">
                    <span className="text-sm font-semibold capitalize">
                      {isDoctor
                        ? '👨‍⚕️ Doctor'
                        : isGroupTherapyPatient
                          ? `🤖 ${patientName}`
                          : '🤖 Patient'}
                    </span>
                    {modelLabel && <p className="break-words text-xs text-muted-foreground"><ModelChatLink provider={turn.model_provider} model={turn.model_name}>{modelLabel}</ModelChatLink></p>}
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    Turn {turn.turn_number} • {formatDate(turn.created_at)}
                  </span>
                </div>
                <TurnSystemPrompts prompts={turn.metadata?.request_system_prompts} />
                {reasoning && (
                  <div className="mb-3 rounded-md border border-dashed border-muted-foreground/30 bg-muted/40 p-3">
                    <p className="text-xs font-medium text-muted-foreground mb-1">{reasoningLabel}</p>
                    <p className="text-xs whitespace-pre-wrap text-foreground/90">{reasoning}</p>
                  </div>
                )}
                {nativeReasoning && <div className="mb-3 rounded-md border border-dashed border-muted-foreground/30 bg-muted/40 p-3"><p className="mb-1 text-xs font-medium text-muted-foreground">Model-provided reasoning</p><p className="whitespace-pre-wrap text-xs">{nativeReasoning}</p></div>}
                <p className="text-sm whitespace-pre-wrap text-foreground">{hasResponse ? turn.response : '(The model returned no answer text.)'}</p>
                <TurnFinishStatus reason={turn.metadata?.finish_reason} />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
