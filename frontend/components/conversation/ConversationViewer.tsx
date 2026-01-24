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

  return (
    <div className="space-y-4">
      {turns.map((turn) => {
        const responseText = turn.response || turn.prompt || '(No content)'
        console.log('[ConversationViewer] Turn:', turn.id, 'speaker:', turn.speaker, 'response_length:', responseText.length)
        
        return (
          <div
            key={turn.id}
            className={cn(
              'rounded-lg border p-4',
              turn.speaker === 'doctor'
                ? 'bg-blue-50 border-blue-200 ml-8'
                : 'bg-gray-50 border-gray-200 mr-8'
            )}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold capitalize">
                {turn.speaker === 'doctor' ? '👨‍⚕️ Doctor' : '🤖 Patient'}
              </span>
              <span className="text-xs text-muted-foreground">
                Turn {turn.turn_number} • {formatDate(turn.created_at)}
              </span>
            </div>
            <p className="text-sm whitespace-pre-wrap">{responseText}</p>
          </div>
        )
      })}
    </div>
  )
}
