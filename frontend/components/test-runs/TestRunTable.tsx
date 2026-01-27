import { useState } from 'react'
import Link from 'next/link'
import { TestRun } from '@/lib/api'
import { DataTable } from '@/components/common/DataTable'
import { StatusBadge } from './StatusBadge'
import { formatDateTime } from '@/lib/utils'
import { MoreVertical, Eye, Trash2, Download } from 'lucide-react'

interface TestRunTableProps {
  testRuns: TestRun[]
  onDelete?: (id: number) => void
}

export function TestRunTable({ testRuns, onDelete }: TestRunTableProps) {
  const [expandedRow, setExpandedRow] = useState<number | null>(null)

  const columns = [
    {
      key: 'id',
      header: 'ID',
    },
    {
      key: 'test_type',
      header: 'Type',
      render: (run: TestRun) => (
        <span className="capitalize font-medium">{run.test_type}</span>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (run: TestRun) => <StatusBadge status={run.status} />,
    },
    {
      key: 'doctor',
      header: 'Doctor Model',
      render: (run: TestRun) => (
        <div className="text-sm">
          <div className="font-medium">{run.doctor_model}</div>
          <div className="text-xs text-muted-foreground">{run.doctor_provider}</div>
        </div>
      ),
    },
    {
      key: 'patient',
      header: 'Patient Model',
      render: (run: TestRun) => (
        <div className="text-sm">
          <div className="font-medium">{run.patient_model}</div>
          <div className="text-xs text-muted-foreground">{run.patient_provider}</div>
        </div>
      ),
    },
    {
      key: 'created_at',
      header: 'Date & Time',
      render: (run: TestRun) => (
        <div className="text-sm">
          {run.created_at ? (
            <div>
              <div className="font-medium">{formatDateTime(run.created_at)}</div>
            </div>
          ) : (
            '-'
          )}
        </div>
      ),
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (run: TestRun) => (
        <div className="flex items-center gap-2">
          <Link
            href={`/test-runs/${run.id}`}
            onClick={(e) => e.stopPropagation()}
            className="rounded p-1 hover:bg-muted"
            title="View Details"
          >
            <Eye className="h-4 w-4" />
          </Link>
          <button
            onClick={(e) => {
              e.stopPropagation()
              onDelete?.(run.id)
            }}
            className="rounded p-1 text-destructive hover:bg-muted"
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      ),
    },
  ]

  return (
    <DataTable
      data={testRuns}
      columns={columns}
      onRowClick={(run) => {
        window.location.href = `/test-runs/${run.id}`
      }}
      emptyMessage="No test runs found"
    />
  )
}
