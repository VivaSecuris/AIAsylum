import { cn } from '@/lib/utils'

interface StatusBadgeProps {
  status: 'pending' | 'running' | 'completed' | 'failed'
  className?: string
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const variants = {
    pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    running: 'bg-blue-100 text-blue-800 border-blue-200',
    completed: 'bg-green-100 text-green-800 border-green-200',
    failed: 'bg-red-100 text-red-800 border-red-200',
  }

  const labels = {
    pending: 'Pending',
    running: 'Running',
    completed: 'Completed',
    failed: 'Failed',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium',
        variants[status],
        className
      )}
    >
      {labels[status]}
    </span>
  )
}
