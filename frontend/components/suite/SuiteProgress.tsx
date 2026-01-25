import { SuiteProgress as SuiteProgressType } from '@/lib/api'
import { ProgressBreakdown } from './ProgressBreakdown'
import { RunningTestsList } from './RunningTestsList'
import { CheckCircle2, Clock, Loader2, XCircle } from 'lucide-react'

interface SuiteProgressProps {
  progress: SuiteProgressType
  suiteName?: string
  suiteStatus: string
}

export function SuiteProgress({ progress, suiteName, suiteStatus }: SuiteProgressProps) {
  const formatTime = (seconds?: number): string => {
    if (!seconds) return 'N/A'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`
    }
    if (minutes > 0) {
      return `${minutes}m ${secs}s`
    }
    return `${secs}s`
  }

  return (
    <div className="space-y-6">
      {/* Header with Progress Bar */}
      <div className="space-y-4">
        {suiteName && (
          <h2 className="text-2xl font-bold">{suiteName}</h2>
        )}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">
              {progress.progress_percentage.toFixed(1)}% Complete
            </span>
            <span className="text-sm text-muted-foreground">
              {progress.completed_runs} / {progress.total_runs} test runs
            </span>
          </div>
          <div className="h-4 w-full rounded-full bg-muted overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${progress.progress_percentage}%` }}
            />
          </div>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="h-4 w-4 text-yellow-600" />
            <span className="text-sm font-medium">Pending</span>
          </div>
          <p className="text-2xl font-bold">{progress.pending_runs}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
            <span className="text-sm font-medium">Running</span>
          </div>
          <p className="text-2xl font-bold">{progress.running_runs}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <span className="text-sm font-medium">Completed</span>
          </div>
          <p className="text-2xl font-bold">{progress.completed_runs}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <XCircle className="h-4 w-4 text-red-600" />
            <span className="text-sm font-medium">Failed</span>
          </div>
          <p className="text-2xl font-bold">{progress.failed_runs}</p>
        </div>
      </div>

      {/* Time Tracking */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border bg-card p-4">
          <div className="text-sm font-medium mb-1">Elapsed Time</div>
          <p className="text-xl font-bold">{formatTime(progress.elapsed_time_seconds)}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <div className="text-sm font-medium mb-1">Estimated Remaining</div>
          <p className="text-xl font-bold">{formatTime(progress.estimated_remaining_seconds)}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <div className="text-sm font-medium mb-1">Avg Time per Run</div>
          <p className="text-xl font-bold">{formatTime(progress.average_time_per_run_seconds)}</p>
        </div>
      </div>

      {/* Currently Running Tests */}
      {progress.running_tests.length > 0 && (
        <RunningTestsList runningTests={progress.running_tests} />
      )}

      {/* Progress Breakdown */}
      <ProgressBreakdown
        progressByModel={progress.progress_by_model}
        progressByTestType={progress.progress_by_test_type}
      />
    </div>
  )
}
