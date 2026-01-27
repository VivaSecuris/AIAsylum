import { ConversationTurn } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import { cn } from '@/lib/utils'

interface ConversationViewerProps {
  turns: ConversationTurn[]
}

export function ConversationViewer({ turns }: ConversationViewerProps) {
  // Debug logging
  console.log('[ConversationViewer] Received turns:', turns?.length || 0, turns)
  if (turns && turns.length > 0) {
    console.log('[ConversationViewer] First turn sample:', {
      id: turns[0].id,
      speaker: turns[0].speaker,
      hasPrompt: !!turns[0].prompt,
      hasResponse: !!turns[0].response,
      promptLength: turns[0].prompt?.length || 0,
      responseLength: turns[0].response?.length || 0,
    })
  }
  
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
        const reasoning = turn.metadata?.reasoning
        const patientName = turn.metadata?.patient_name
        const patientId = turn.metadata?.patient_id
        const patientModel = turn.metadata?.patient_model
        console.log('[ConversationViewer] Turn:', turn.id, 'speaker:', turn.speaker, 'prompt:', turn.prompt?.substring(0, 50), 'response:', turn.response?.substring(0, 50))
        
        // Determine if this is a group therapy patient response
        const isGroupTherapyPatient = turn.speaker === 'patient' && patientName !== undefined
        
        // Get color for this patient (if group therapy)
        const patientColorIndex = patientId !== undefined ? (patientId % patientColors.length) : 0
        const patientColorClass = isGroupTherapyPatient ? patientColors[patientColorIndex] : 'bg-gray-50 border-gray-200 dark:bg-gray-900 dark:border-gray-700'
        
        const isDoctor = turn.speaker === 'doctor'
        const hasPrompt = turn.prompt && turn.prompt.trim().length > 0
        const hasResponse = turn.response && turn.response.trim().length > 0
        
        // The main content is the speaker's response
        // The prompt is what they're responding to (context)
        
        return (
          <div key={turn.id} className="space-y-2">
            {/* Context/Prompt (what the speaker is responding to) - shown in smaller, muted style */}
            {hasPrompt && (
              <div className="ml-4 text-xs text-muted-foreground italic border-l-2 border-muted pl-2">
                <span className="font-medium">Context: </span>
                <span>{turn.prompt.substring(0, 200)}{turn.prompt.length > 200 ? '...' : ''}</span>
              </div>
            )}
            
            {/* Main Message - The speaker's response */}
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
                {reasoning && (
                  <div className="mb-3 rounded border-l-4 border-amber-400 bg-amber-50/50 dark:bg-amber-950/50 p-3">
                    <div className="mb-1 text-xs font-semibold text-amber-800 dark:text-amber-200">
                      💭 Internal Reasoning (CoT/ReACT)
                    </div>
                    <pre className="text-xs text-amber-900 dark:text-amber-100 whitespace-pre-wrap font-mono">
                      {reasoning}
                    </pre>
                  </div>
                )}
                <p className="text-sm whitespace-pre-wrap text-foreground">{turn.response}</p>
              </div>
            )}
            
            {/* Fallback if neither prompt nor response exists */}
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
