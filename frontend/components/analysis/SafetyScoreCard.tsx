import { Assessment } from '@/lib/api'
import { cn } from '@/lib/utils'

interface SafetyScoreCardProps {
  assessment: Assessment
  className?: string
}

export function SafetyScoreCard({ assessment, className }: SafetyScoreCardProps) {
  const score = assessment.safety_score || assessment.overall_score || 0
  const scorePercent = (score * 100).toFixed(1)

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600'
    if (score >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className={cn('rounded-lg border bg-card p-6', className)}>
      <h3 className="text-sm font-medium text-muted-foreground mb-2">Safety Score</h3>
      <div className="flex items-baseline gap-2">
        <span className={cn('text-4xl font-bold', getScoreColor(score))}>
          {scorePercent}%
        </span>
        <span className="text-sm text-muted-foreground">/ 100%</span>
      </div>
      {assessment.scores && (
        <div className="mt-4 space-y-2">
          {Object.entries(assessment.scores).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between text-sm">
              <span className="capitalize text-muted-foreground">{key.replace('_', ' ')}</span>
              <span className="font-medium">{(value * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
