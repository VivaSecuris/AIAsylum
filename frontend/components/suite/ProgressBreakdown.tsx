interface ProgressBreakdownProps {
  progressByModel: Record<string, {
    total: number
    completed: number
    failed: number
    running: number
    pending: number
  }>
  progressByTestType: Record<string, {
    total: number
    completed: number
    failed: number
    running: number
    pending: number
  }>
}

export function ProgressBreakdown({ progressByModel, progressByTestType }: ProgressBreakdownProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Progress by Model */}
      <div className="rounded-lg border bg-card p-4">
        <h3 className="text-sm font-semibold mb-4">Progress by Model</h3>
        <div className="space-y-3">
          {Object.entries(progressByModel).map(([model, stats]) => {
            const progress = stats.total > 0 ? (stats.completed / stats.total) * 100 : 0
            return (
              <div key={model} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">{model}</span>
                  <span className="text-muted-foreground">
                    {stats.completed}/{stats.total}
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="flex gap-4 text-xs text-muted-foreground">
                  <span>✓ {stats.completed}</span>
                  {stats.running > 0 && <span>⟳ {stats.running}</span>}
                  {stats.failed > 0 && <span className="text-red-600">✗ {stats.failed}</span>}
                  {stats.pending > 0 && <span>⏱ {stats.pending}</span>}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Progress by Test Type */}
      <div className="rounded-lg border bg-card p-4">
        <h3 className="text-sm font-semibold mb-4">Progress by Test Type</h3>
        <div className="space-y-3">
          {Object.entries(progressByTestType).map(([testType, stats]) => {
            const progress = stats.total > 0 ? (stats.completed / stats.total) * 100 : 0
            return (
              <div key={testType} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium capitalize">{testType}</span>
                  <span className="text-muted-foreground">
                    {stats.completed}/{stats.total}
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="flex gap-4 text-xs text-muted-foreground">
                  <span>✓ {stats.completed}</span>
                  {stats.running > 0 && <span>⟳ {stats.running}</span>}
                  {stats.failed > 0 && <span className="text-red-600">✗ {stats.failed}</span>}
                  {stats.pending > 0 && <span>⏱ {stats.pending}</span>}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
