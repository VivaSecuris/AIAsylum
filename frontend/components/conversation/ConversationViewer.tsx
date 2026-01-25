import { ConversationTurn } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import { cn } from '@/lib/utils'

interface ConversationViewerProps {
  turns: ConversationTurn[]
}

export function ConversationViewer({ turns }: ConversationViewerProps) {
  // Debug logging
  console.log('[ConversationViewer] Received turns:', turns?.length || 0, turns)
  
  if (!turns || turns.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
        No conversation turns available
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
        const responseText = turn.response || turn.prompt || '(No content)'
        const reasoning = turn.metadata?.reasoning
        const patientName = turn.metadata?.patient_name
        const patientId = turn.metadata?.patient_id
        const patientModel = turn.metadata?.patient_model
        console.log('[ConversationViewer] Turn:', turn.id, 'speaker:', turn.speaker, 'response_length:', responseText.length, 'reasoning_length:', reasoning?.length || 0)
        
        // Determine if this is a group therapy patient response
        const isGroupTherapyPatient = turn.speaker === 'patient' && patientName !== undefined
        
        // Get color for this patient (if group therapy)
        const patientColorIndex = patientId !== undefined ? (patientId % patientColors.length) : 0
        const patientColorClass = isGroupTherapyPatient ? patientColors[patientColorIndex] : 'bg-gray-50 border-gray-200'
        
        return (
          <div
            key={turn.id}
            className={cn(
              'rounded-lg border p-4',
              turn.speaker === 'doctor'
                ? 'bg-blue-50 border-blue-200 ml-8'
                : patientColorClass + ' mr-8'
            )}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold capitalize">
                {turn.speaker === 'doctor' 
                  ? '👨‍⚕️ Doctor' 
                  : isGroupTherapyPatient
                    ? `🤖 ${patientName}${patientModel ? ` (${patientModel})` : ''}`
                    : '🤖 Patient'}
              </span>
              <span className="text-xs text-muted-foreground">
                Turn {turn.turn_number} • {formatDate(turn.created_at)}
              </span>
            </div>
            {reasoning && (
              <div className="mb-3 rounded border-l-4 border-amber-400 bg-amber-50/50 p-3">
                <div className="mb-1 text-xs font-semibold text-amber-800">
                  💭 Internal Reasoning (CoT/ReACT)
                </div>
                <pre className="text-xs text-amber-900 whitespace-pre-wrap font-mono">
                  {reasoning}
                </pre>
              </div>
            )}
            <p className="text-sm whitespace-pre-wrap">{responseText}</p>
          </div>
        )
      })}
    </div>
  )
}
