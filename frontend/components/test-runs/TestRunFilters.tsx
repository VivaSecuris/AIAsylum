import { useState } from 'react'
import { TestRun } from '@/lib/api'

interface TestRunFiltersProps {
  onFilterChange: (filters: {
    test_type?: string
    status?: string
    search?: string
  }) => void
}

export function TestRunFilters({ onFilterChange }: TestRunFiltersProps) {
  const [testType, setTestType] = useState<string>('')
  const [status, setStatus] = useState<string>('')
  const [search, setSearch] = useState<string>('')

  const handleChange = (key: string, value: string) => {
    const newFilters: any = { test_type: testType, status, search }
    newFilters[key] = value
    if (key === 'test_type') setTestType(value)
    if (key === 'status') setStatus(value)
    if (key === 'search') setSearch(value)
    onFilterChange(newFilters)
  }

  return (
    <div className="flex flex-wrap gap-4 rounded-lg border bg-card p-4">
      <div className="flex-1 min-w-[200px]">
        <input
          type="text"
          placeholder="Search by name or model..."
          value={search}
          onChange={(e) => handleChange('search', e.target.value)}
          className="w-full rounded border px-3 py-2 text-sm"
        />
      </div>
      <select
        value={testType}
        onChange={(e) => handleChange('test_type', e.target.value)}
        className="rounded border px-3 py-2 text-sm"
      >
        <option value="">All Test Types</option>
        <option value="one_shot">One-Shot</option>
        <option value="multi_shot">Multi-Shot</option>
        <option value="conversation">Conversation</option>
        <option value="group_therapy">Group Therapy</option>
        <option value="benchmark">Benchmark</option>
        <option value="analysis">Analysis</option>
        <option value="scenario">Scenario (legacy)</option>
        <option value="adversarial">Adversarial (legacy)</option>
      </select>
      <select
        value={status}
        onChange={(e) => handleChange('status', e.target.value)}
        className="rounded border px-3 py-2 text-sm"
      >
        <option value="">All Statuses</option>
        <option value="pending">Pending</option>
        <option value="running">Running</option>
        <option value="paused">Paused</option>
        <option value="completed">Completed</option>
        <option value="failed">Failed</option>
      </select>
      {(testType || status || search) && (
        <button
          onClick={() => {
            setTestType('')
            setStatus('')
            setSearch('')
            onFilterChange({})
          }}
          className="rounded border px-3 py-2 text-sm hover:bg-muted"
        >
          Clear
        </button>
      )}
    </div>
  )
}
