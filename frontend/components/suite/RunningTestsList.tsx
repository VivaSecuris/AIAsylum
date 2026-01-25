import { Loader2 } from 'lucide-react'

interface RunningTest {
  id: number
  model: string
  test_type: string
  benchmark?: string
  started_at?: string
}

interface RunningTestsListProps {
  runningTests: RunningTest[]
}

export function RunningTestsList({ runningTests }: RunningTestsListProps) {
  const getElapsedTime = (startedAt?: string): string => {
    if (!startedAt) return 'N/A'
    const start = new Date(startedAt)
    const now = new Date()
    const seconds = Math.floor((now.getTime() - start.getTime()) / 1000)
    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60
    if (minutes > 0) {
      return `${minutes}m ${secs}s`
    }
    return `${secs}s`
  }

  return (
    <div className="rounded-lg border bg-card p-4">
      <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
        <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
        Currently Running ({runningTests.length})
      </h3>
      <div className="space-y-2">
        {runningTests.map((test) => (
          <div
            key={test.id}
            className="flex items-center justify-between rounded border bg-muted/30 p-3"
          >
            <div className="flex items-center gap-3">
              <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
              <div>
                <div className="text-sm font-medium">
                  {test.model} - {test.benchmark || test.test_type}
                </div>
                <div className="text-xs text-muted-foreground">
                  Test Run #{test.id}
                </div>
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              {getElapsedTime(test.started_at)}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
