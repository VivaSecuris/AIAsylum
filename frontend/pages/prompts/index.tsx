import { useState } from 'react'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { usePrompts, useDeletePrompt } from '@/lib/hooks'
import { Prompt } from '@/lib/api'
import { Plus, Edit, Trash2, Search, Tag } from 'lucide-react'
import Link from 'next/link'
import { toast } from '@/lib/toast'
import { formatDate } from '@/lib/utils'

export default function PromptsPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('')
  const [promptTypeFilter, setPromptTypeFilter] = useState<string>('')
  const [targetFilter, setTargetFilter] = useState<string>('')
  const { data: prompts = [], isLoading } = usePrompts()
  const deletePrompt = useDeletePrompt()

  const filteredPrompts = prompts.filter((prompt) => {
    const matchesSearch =
      prompt.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      prompt.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      prompt.prompt_text.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = !categoryFilter || prompt.category === categoryFilter
    const matchesPromptType = !promptTypeFilter || prompt.prompt_type === promptTypeFilter
    const matchesTarget = !targetFilter || prompt.target === targetFilter
    return matchesSearch && matchesCategory && matchesPromptType && matchesTarget
  })

  // Sort categories: essential jailbreak categories first, then others, forbidden_question last
  const categoryOrder = [
    'adversarial',  // Jailbreak prompts
    'jailbreak',    // Alternative jailbreak category name
    'red_team',     // Red team prompts
    'adversarial_prompt', // Another jailbreak variant
    'roleplay',     // Roleplay jailbreaks
    'direct_bypass', // Direct bypass techniques
    'indirect_bypass', // Indirect bypass techniques
    'encoding',     // Encoding-based jailbreaks
    'conversation',
    'scenario',
    'reasoning',
    'safety',
    'forbidden_question',  // Always last
  ]
  
  const allCategories = Array.from(new Set(prompts.map((p) => p.category).filter(Boolean)))
  
  // Build categories list: show all ordered categories that exist in data, then others, then forbidden last
  const orderedCategories = categoryOrder.filter(cat => allCategories.includes(cat))
  const otherCategories = allCategories.filter(cat => !categoryOrder.includes(cat)).sort()
  
  // Ensure forbidden_question is always last if it exists
  const forbiddenIndex = orderedCategories.indexOf('forbidden_question')
  if (forbiddenIndex !== -1) {
    orderedCategories.splice(forbiddenIndex, 1)
  }
  
  const categories = [
    ...orderedCategories,
    ...otherCategories,
    ...(allCategories.includes('forbidden_question') ? ['forbidden_question'] : []),
  ]

  const handleDelete = async (id: number) => {
    if (confirm('Are you sure you want to delete this prompt?')) {
      try {
        await deletePrompt.mutateAsync(id)
        toast.success('Prompt deleted successfully')
      } catch (error) {
        toast.error('Failed to delete prompt')
      }
    }
  }

  if (isLoading) {
    return (
      <Layout>
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" />
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Prompt Library</h1>
          <Link
            href="/prompts/create"
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Plus className="h-4 w-4" />
            Create Prompt
          </Link>
        </div>

        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px] relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search prompts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full rounded-lg border pl-10 pr-4 py-2 text-sm"
            />
          </div>
          <select
            value={promptTypeFilter}
            onChange={(e) => setPromptTypeFilter(e.target.value)}
            className="rounded-lg border px-4 py-2 text-sm"
          >
            <option value="">All Types</option>
            <option value="test_prompt">Test Prompts</option>
            <option value="system_prompt">System Prompts</option>
          </select>
          {promptTypeFilter === 'system_prompt' && (
            <select
              value={targetFilter}
              onChange={(e) => setTargetFilter(e.target.value)}
              className="rounded-lg border px-4 py-2 text-sm"
            >
              <option value="">All Targets</option>
              <option value="doctor">Doctor</option>
              <option value="patient">Patient</option>
            </select>
          )}
          {categories.length > 0 && (
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="rounded-lg border px-4 py-2 text-sm"
            >
              <option value="">All Categories</option>
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat === 'adversarial' ? 'Jailbreak Prompts' : 
                   cat === 'forbidden_question' ? 'Forbidden Questions' :
                   cat.charAt(0).toUpperCase() + cat.slice(1).replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          )}
        </div>

        <div className="text-sm text-muted-foreground">
          Showing {filteredPrompts.length} of {prompts.length} prompts
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredPrompts.map((prompt) => (
            <PromptCard key={prompt.id} prompt={prompt} onDelete={handleDelete} />
          ))}
        </div>

        {filteredPrompts.length === 0 && (
          <div className="rounded-lg border bg-card p-8 text-center">
            <p className="text-muted-foreground">No prompts found. Create your first prompt to get started.</p>
          </div>
        )}
      </div>
    </Layout>
  )
}

function PromptCard({ prompt, onDelete }: { prompt: Prompt; onDelete: (id: number) => void }) {
  return (
    <div className="rounded-lg border bg-card p-6 space-y-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="font-semibold text-lg mb-1">{prompt.name}</h3>
          {prompt.description && <p className="text-sm text-muted-foreground line-clamp-2">{prompt.description}</p>}
        </div>
        <div className="flex gap-2">
          <Link
            href={`/prompts/${prompt.id}/edit`}
            className="p-2 hover:bg-muted rounded"
            title="Edit"
          >
            <Edit className="h-4 w-4" />
          </Link>
          <button
            onClick={() => onDelete(prompt.id)}
            className="p-2 hover:bg-muted rounded text-destructive"
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="text-sm text-muted-foreground space-y-1">
        <p className="line-clamp-3">{prompt.prompt_text}</p>
      </div>

      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span className={`px-2 py-1 rounded ${
          prompt.prompt_type === 'system_prompt' 
            ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' 
            : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
        }`}>
          {prompt.prompt_type === 'system_prompt' ? 'System' : 'Test'}
          {prompt.target && ` - ${prompt.target}`}
        </span>
        {prompt.category && (
          <span className="flex items-center gap-1">
            <Tag className="h-3 w-3" />
            {prompt.category}
          </span>
        )}
        <span>Used {prompt.usage_count} times</span>
        {prompt.created_at && <span>{formatDate(prompt.created_at)}</span>}
      </div>

      {prompt.tags && prompt.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {prompt.tags.map((tag, idx) => (
            <span key={idx} className="text-xs bg-muted px-2 py-1 rounded">
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
