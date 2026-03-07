import { ConversationTurn } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import { cn } from '@/lib/utils'

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
      {turns.map((turn) => {
        const patientName = turn.metadata?.patient_name
        const patientId = turn.metadata?.patient_id
        const patientModel = turn.metadata?.patient_model

        const isGroupTherapyPatient = turn.speaker === 'patient' && patientName !== undefined
        const patientColorIndex = patientId !== undefined ? (patientId % patientColors.length) : 0
        const patientColorClass = isGroupTherapyPatient ? patientColors[patientColorIndex] : 'bg-gray-50 border-gray-200 dark:bg-gray-900 dark:border-gray-700'

        const isDoctor = turn.speaker === 'doctor'
        const hasPrompt = turn.prompt && turn.prompt.trim().length > 0
        const hasResponse = turn.response && turn.response.trim().length > 0

        // Context = what this speaker is replying to (the other party's message). Label with the other speaker.
        const replyingToLabel = isDoctor ? '🤖 Patient' : '👨‍⚕️ Doctor'

        return (
          <div key={turn.id} className="space-y-1">
            {/* Compact context: what this AI is responding to (other speaker's message) */}
            {hasPrompt && (
              <div className="ml-4 text-xs text-muted-foreground italic border-l-2 border-muted pl-2">
                <span className="font-medium">Replying to {replyingToLabel}: </span>
                <span>{turn.prompt.length > 180 ? `${turn.prompt.substring(0, 180)}…` : turn.prompt}</span>
              </div>
            )}

            {/* Main message: this speaker's response (one bubble per turn, no duplicate) */}
            {hasResponse && (
              <div
                className={cn(
                  'rounded-lg border p-4',
                  isDoctor
                    ? 'bg-blue-50 border-blue-200 dark:bg-blue-950 dark:border-blue-800 ml-8'
                    : patientColorClass + ' mr-8'
                )}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold capitalize">
                    {isDoctor
                      ? '👨‍⚕️ Doctor'
                      : isGroupTherapyPatient
                        ? `🤖 ${patientName}${patientModel ? ` (${patientModel})` : ''}`
                        : '🤖 Patient'}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    Turn {turn.turn_number} • {formatDate(turn.created_at)}
                  </span>
                </div>
                <p className="text-sm whitespace-pre-wrap text-foreground">{turn.response}</p>
              </div>
            )}

            {!hasPrompt && !hasResponse && (
              <div className="rounded-lg border bg-card p-4 text-center text-muted-foreground">
                Turn {turn.turn_number}: No content available
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
