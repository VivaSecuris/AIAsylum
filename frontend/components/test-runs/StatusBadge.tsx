import { cn } from '@/lib/utils'
import type { TestRunStatus, SuiteStatus } from '@/lib/api'

type Status = TestRunStatus | SuiteStatus

interface StatusBadgeProps {
  status: Status
  className?: string
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const variants: Record<Status, string> = {
    pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    running: 'bg-blue-100 text-blue-800 border-blue-200',
    paused: 'bg-purple-100 text-purple-800 border-purple-200',
    completed: 'bg-green-100 text-green-800 border-green-200',
    failed: 'bg-red-100 text-red-800 border-red-200',
    partially_failed: 'bg-orange-100 text-orange-800 border-orange-200',
  }

  const labels: Record<Status, string> = {
    pending: 'Pending',
    running: 'Running',
    paused: 'Paused',
    completed: 'Completed',
    failed: 'Failed',
    partially_failed: 'Partially Failed',
  }

  // The status columns are plain strings in the DB, so tolerate a value outside the known set
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium',
        variants[status] ?? 'bg-gray-100 text-gray-800 border-gray-200',
        className
      )}
    >
      {labels[status] ?? status}
    </span>
  )
}
