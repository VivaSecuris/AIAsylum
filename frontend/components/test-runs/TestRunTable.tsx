import { useState } from 'react'
import Link from 'next/link'
import { TestRun } from '@/lib/api'
import { DataTable } from '@/components/common/DataTable'
import { StatusBadge } from './StatusBadge'
import { formatDateTime } from '@/lib/utils'
import { TEST_TYPES } from '@/lib/create-test-config'
import { MoreVertical, Eye, Trash2, Download } from 'lucide-react'
import { shortModelRef, testRunModelRef } from '@/lib/test-run-display'
import { ModelChatLink } from '@/components/models/ModelChatLink'

interface TestRunTableProps {
  testRuns: TestRun[]
  allTestRuns?: TestRun[]
  onDelete?: (id: number) => void
}

export function TestRunTable({ testRuns, allTestRuns = testRuns, onDelete }: TestRunTableProps) {
  const [expandedRow, setExpandedRow] = useState<number | null>(null)
  const activeCampaignIds = new Set(allTestRuns
    .filter((run) => ['pending', 'running', 'paused'].includes(run.status))
    .map((run) => run.meta_data?.benchmark_campaign?.id)
    .filter((id): id is string => typeof id === 'string'))

  const columns = [
    {
      key: 'id',
      header: 'ID',
    },
    {
      key: 'name',
      header: 'Name',
      render: (run: TestRun) => (
        <div className="text-sm">
          <div className="font-medium break-words">{run.meta_data?.name || '—'}</div>
          {run.suite_name && <div className="text-xs text-muted-foreground">Suite: {run.suite_name}</div>}
        </div>
      ),
    },
    {
      key: 'test_type',
      header: 'Type',
      render: (run: TestRun) => (
        <span className="font-medium">{TEST_TYPES.find((t) => t.value === run.test_type)?.label ?? (run.test_type === 'analysis' ? 'Analysis' : run.test_type.replace(/_/g, ' '))}</span>
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
      render: (run: TestRun) => run.test_type === 'benchmark' ? <span title="Objective benchmark; no doctor model">—</span> : (
        <div className="text-sm">
          <ModelChatLink className="font-medium" provider={run.doctor_provider} model={run.doctor_model} />
          <div className="text-xs text-muted-foreground">{run.doctor_provider}</div>
        </div>
      ),
    },
    {
      key: 'patient',
      header: 'Patient Model',
      render: (run: TestRun) => (
        <div className="text-sm">
          <ModelChatLink className="font-medium break-words" provider={run.patient_provider} model={run.patient_model}>{shortModelRef(testRunModelRef(run))}</ModelChatLink>
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
              <div className="font-medium whitespace-nowrap">{formatDateTime(run.created_at)}</div>
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
          {!activeCampaignIds.has(run.meta_data?.benchmark_campaign?.id) && <button
            onClick={(e) => {
              e.stopPropagation()
              onDelete?.(run.id)
            }}
            className="rounded p-1 text-destructive hover:bg-muted"
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </button>}
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
