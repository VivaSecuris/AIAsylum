import { useState, useMemo } from 'react'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { usePrompts, useDeletePrompt } from '@/lib/hooks'
import { Prompt } from '@/lib/api'
import { Plus, Edit, Trash2, Search, Tag, Play } from 'lucide-react'
import Link from 'next/link'
import { toast } from '@/lib/toast'
import { formatDate, getPromptDisplayName } from '@/lib/utils'

export default function PromptsPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('') // Start with all prompts, user can filter
  const [promptTypeFilter, setPromptTypeFilter] = useState<string>('')
  const [targetFilter, setTargetFilter] = useState<string>('')
  const [techniqueFilter, setTechniqueFilter] = useState<string>('')
  const [showJailbreaks, setShowJailbreaks] = useState(false) // Hide jailbreak prompts by default
  const [showBenchmarks, setShowBenchmarks] = useState(false) // Hide benchmark prompts by default
  
  // Fetch prompts - increase limit to get all prompts
  // Pass category filter only if explicitly set by user
  const promptParams: { category?: string; limit: number } = { limit: 2000 }
  if (categoryFilter && categoryFilter !== '') {
    promptParams.category = categoryFilter
  }
  const { data: prompts = [], isLoading, error } = usePrompts(promptParams)
  const deletePrompt = useDeletePrompt()

  // Helper function to check if a prompt is a jailbreak prompt
  const isJailbreakPrompt = (prompt: Prompt) => {
    const jailbreakCategories = ['adversarial', 'jailbreak', 'red_team', 'adversarial_prompt', 
                                 'roleplay', 'direct_bypass', 'indirect_bypass', 'encoding']
    return jailbreakCategories.includes(prompt.category || '') ||
           (prompt.tags && prompt.tags.some(tag => tag === 'jailbreak' || tag === 'adversarial'))
  }
  
  // Helper function to check if a prompt is a benchmark prompt
  const isBenchmarkPrompt = (prompt: Prompt) => {
    return prompt.category === 'benchmark' ||
           (prompt.tags && prompt.tags.some(tag => tag === 'benchmark')) ||
           (prompt.metadata && prompt.metadata.benchmark)
  }

  // Extract techniques from jailbreak prompts for filtering (use all prompts for accurate counts)
  const techniques = Array.from(
    new Set(
      prompts
        .filter(p => isJailbreakPrompt(p) && p.metadata?.jailbreak_technique)
        .map(p => p.metadata.jailbreak_technique)
        .filter(Boolean)
    )
  ).sort()

  // Count prompts by technique (use all prompts for accurate counts)
  const techniqueCounts: Record<string, number> = {}
  prompts
    .filter(p => isJailbreakPrompt(p) && p.metadata?.jailbreak_technique)
    .forEach(p => {
      const tech = p.metadata.jailbreak_technique
      techniqueCounts[tech] = (techniqueCounts[tech] || 0) + 1
    })
  
  // Filter and randomize prompts based on visibility settings
  const visiblePrompts = useMemo(() => {
    // First filter based on visibility settings
    let filtered = prompts.filter((prompt) => {
      // Completely exclude forbidden questions from prompt library
      if (prompt.category === 'forbidden_question') {
        return false
      }
      
      // Hide jailbreak prompts unless explicitly requested
      if (isJailbreakPrompt(prompt) && !showJailbreaks) {
        return false
      }
      
      // Hide benchmark prompts unless explicitly requested
      if (isBenchmarkPrompt(prompt) && !showBenchmarks) {
        return false
      }
      
      return true
    })
    
    // Randomize and deduplicate jailbreak/benchmark prompts when showing them
    if (showJailbreaks || showBenchmarks) {
      const jailbreakPrompts = filtered.filter(isJailbreakPrompt)
      const benchmarkPrompts = filtered.filter(isBenchmarkPrompt)
      const otherPrompts = filtered.filter(p => !isJailbreakPrompt(p) && !isBenchmarkPrompt(p))
      
      // Shuffle arrays for randomization
      const shuffle = <T,>(array: T[]): T[] => {
        const shuffled = [...array]
        for (let i = shuffled.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
        }
        return shuffled
      }
      
      // Deduplicate by prompt_text hash (simple approach)
      const deduplicate = (prompts: Prompt[]): Prompt[] => {
        const seen = new Set<string>()
        return prompts.filter(p => {
          const key = p.prompt_text.trim().toLowerCase()
          if (seen.has(key)) return false
          seen.add(key)
          return true
        })
      }
      
      const randomizedJailbreaks = showJailbreaks ? deduplicate(shuffle(jailbreakPrompts)) : []
      const randomizedBenchmarks = showBenchmarks ? deduplicate(shuffle(benchmarkPrompts)) : []
      
      filtered = [...otherPrompts, ...randomizedJailbreaks, ...randomizedBenchmarks]
    }
    
    return filtered
  }, [prompts, showJailbreaks, showBenchmarks])
  
  const filteredPrompts = visiblePrompts.filter((prompt) => {
    // If category filter is set, the API should already filter, but do client-side check too
    // (in case API filtering doesn't work or for additional safety)
    const matchesCategory = !categoryFilter || categoryFilter === '' || prompt.category === categoryFilter
    
    const matchesSearch =
      !searchTerm ||
      prompt.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      getPromptDisplayName(prompt).toLowerCase().includes(searchTerm.toLowerCase()) ||
      prompt.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      prompt.prompt_text.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesPromptType = !promptTypeFilter || prompt.prompt_type === promptTypeFilter
    const matchesTarget = !targetFilter || prompt.target === targetFilter
    
    // Filter by technique for jailbreak prompts
    const matchesTechnique = !techniqueFilter || 
      (prompt.category === 'adversarial' && 
       prompt.metadata?.jailbreak_technique === techniqueFilter)
    
    return matchesCategory && matchesSearch && matchesPromptType && matchesTarget && matchesTechnique
  })

  // Count prompts by category (from loaded data)
  const jailbreakCount = prompts.filter(isJailbreakPrompt).length
  const benchmarkCount = prompts.filter(isBenchmarkPrompt).length
  const otherCount = prompts.filter(p => !isJailbreakPrompt(p) && !isBenchmarkPrompt(p) && p.category !== 'forbidden_question').length
  const totalLoaded = prompts.length
  const visibleCount = visiblePrompts.length

  // Sort filtered prompts: jailbreak prompts first, then by technique, then others
  const sortedPrompts = [...filteredPrompts].sort((a, b) => {
    // Jailbreak prompts first
    if (a.category === 'adversarial' && b.category !== 'adversarial') return -1
    if (a.category !== 'adversarial' && b.category === 'adversarial') return 1
    
    // Within jailbreak prompts, sort by technique
    if (a.category === 'adversarial' && b.category === 'adversarial') {
      const techA = a.metadata?.jailbreak_technique || 'unknown'
      const techB = b.metadata?.jailbreak_technique || 'unknown'
      if (techA !== techB) return techA.localeCompare(techB)
    }
    
    // Then by name
    return a.name.localeCompare(b.name)
  })

  // Sort categories: regular categories first, then jailbreak/benchmark categories
  // Forbidden questions are completely excluded - they should only be used AFTER jailbreak is achieved
  const categoryOrder = [
    'conversation',
    'scenario',
    'reasoning',
    'safety',
    'adversarial',  // Jailbreak prompts (hidden by default)
    'jailbreak',    // Alternative jailbreak category name
    'red_team',     // Red team prompts
    'adversarial_prompt', // Another jailbreak variant
    'roleplay',     // Roleplay jailbreaks
    'direct_bypass', // Direct bypass techniques
    'indirect_bypass', // Indirect bypass techniques
    'encoding',     // Encoding-based jailbreaks
    'benchmark',    // Benchmark prompts (hidden by default)
  ]
  
  // Only show categories from visible prompts
  const allCategories = Array.from(new Set(visiblePrompts.map((p) => p.category).filter(Boolean)))
  
  // Exclude forbidden_question from categories list - they're not for general use
  // Forbidden questions should only be used AFTER a jailbreak is achieved
  const filteredCategories = allCategories.filter(cat => cat !== 'forbidden_question')
  
  // Build categories list: show all ordered categories that exist in data, then others
  const orderedCategories = categoryOrder.filter(cat => filteredCategories.includes(cat))
  const otherCategories = filteredCategories.filter(cat => !categoryOrder.includes(cat)).sort()
  
  const categories = [
    ...orderedCategories,
    ...otherCategories,
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
          <div className="text-center">
            <LoadingSpinner size="lg" />
            <p className="mt-4 text-sm text-muted-foreground">Loading prompts...</p>
          </div>
        </div>
      </Layout>
    )
  }

  // Debug info (remove in production)
  if (process.env.NODE_ENV === 'development') {
    console.log('Prompts loaded:', prompts.length, 'Category filter:', categoryFilter, 'Filtered:', sortedPrompts.length)
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div>
              <h1 className="text-3xl font-bold">Prompt Library</h1>
              <p className="text-sm text-muted-foreground mt-1">
                {totalLoaded > 0 ? (
                  <>
                    {visibleCount} visible prompts
                    {!showJailbreaks && jailbreakCount > 0 && ` • ${jailbreakCount} jailbreak prompts hidden`}
                    {!showBenchmarks && benchmarkCount > 0 && ` • ${benchmarkCount} benchmark prompts hidden`}
                    {categoryFilter === 'adversarial' && techniqueFilter && ` • Filtered by: ${techniqueFilter}`}
                  </>
                ) : error ? (
                  <span className="text-destructive">Error loading prompts. Check API connection.</span>
                ) : (
                  'No prompts loaded - check if API is running and prompts are imported'
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* Toggle buttons for jailbreak and benchmark prompts */}
            {jailbreakCount > 0 && (
              <button
                onClick={() => {
                  setShowJailbreaks(!showJailbreaks)
                  // Clear category filter when toggling to avoid conflicts
                  if (!showJailbreaks) {
                    setCategoryFilter('')
                    setTechniqueFilter('')
                  }
                }}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  showJailbreaks
                    ? 'bg-orange-500 text-white hover:bg-orange-600'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                {showJailbreaks ? 'Hide' : 'Show'} Jailbreaks ({jailbreakCount})
              </button>
            )}
            {benchmarkCount > 0 && (
              <button
                onClick={() => {
                  setShowBenchmarks(!showBenchmarks)
                  // Clear category filter when toggling to avoid conflicts
                  if (!showBenchmarks) {
                    setCategoryFilter('')
                  }
                }}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  showBenchmarks
                    ? 'bg-purple-500 text-white hover:bg-purple-600'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                {showBenchmarks ? 'Hide' : 'Show'} Benchmarks ({benchmarkCount})
              </button>
            )}
            <Link
              href="/prompts/create"
              className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              <Plus className="h-4 w-4" />
              Create Prompt
            </Link>
          </div>
        </div>

        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px] relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search prompts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full rounded-md border border-input bg-background pl-10 pr-4 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            />
          </div>
          <select
            value={promptTypeFilter}
            onChange={(e) => setPromptTypeFilter(e.target.value)}
            className="rounded-md border border-input bg-background px-4 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            <option value="">All Types</option>
            <option value="test_prompt">Test Prompts</option>
            <option value="system_prompt">System Prompts</option>
          </select>
          {promptTypeFilter === 'system_prompt' && (
            <select
              value={targetFilter}
              onChange={(e) => setTargetFilter(e.target.value)}
              className="rounded-md border border-input bg-background px-4 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <option value="">All Targets</option>
              <option value="doctor">Doctor</option>
              <option value="patient">Patient</option>
            </select>
          )}
          {categories.length > 0 && (
            <select
              value={categoryFilter}
              onChange={(e) => {
                setCategoryFilter(e.target.value)
                // Clear technique filter when changing category
                if (e.target.value !== 'adversarial') {
                  setTechniqueFilter('')
                }
              }}
              className="rounded-md border border-input bg-background px-4 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <option value="">All Categories ({totalLoaded} total)</option>
              {categories.map((cat) => {
                const count = visiblePrompts.filter(p => p.category === cat).length
                return (
                  <option key={cat} value={cat}>
                    {cat === 'adversarial' ? `Jailbreak Prompts (${count})` : 
                     cat === 'benchmark' ? `Benchmark Prompts (${count})` :
                     cat === 'forbidden_question' ? 'Forbidden Questions' :
                     `${cat.charAt(0).toUpperCase() + cat.slice(1).replace(/_/g, ' ')} (${count})`}
                  </option>
                )
              })}
            </select>
          )}
          {categoryFilter === 'adversarial' && techniques.length > 0 && (
            <select
              value={techniqueFilter}
              onChange={(e) => setTechniqueFilter(e.target.value)}
              className="rounded-md border border-input bg-background px-4 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <option value="">All Techniques</option>
              {techniques.map((tech) => {
                const count = prompts.filter(
                  p => p.category === 'adversarial' && p.metadata?.jailbreak_technique === tech
                ).length
                return (
                  <option key={tech} value={tech}>
                    {tech.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} ({count})
                  </option>
                )
              })}
            </select>
          )}
        </div>

        {/* Workflow info and technique summary for jailbreak prompts */}
        {showJailbreaks && (categoryFilter === 'adversarial' || categoryFilter === '') && (
          <div className="space-y-4">
            <div className="rounded-lg border bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800 p-4">
              <div className="flex items-start gap-3">
                <div className="flex-1">
                  <h3 className="font-semibold text-sm mb-1">Jailbreak Testing Workflow</h3>
                  <p className="text-xs text-muted-foreground">
                    <strong>Step 1:</strong> Use these jailbreak prompts to attempt to bypass model safety guardrails.
                    <br />
                    <strong>Step 2:</strong> If jailbreak is successful, then use forbidden questions to test what the model will do.
                    <br />
                    <span className="text-muted-foreground/80">
                      Prompts are randomized and deduplicated for uniqueness. Forbidden questions are hidden from this library - they're only available after a jailbreak is detected.
                    </span>
                  </p>
                </div>
              </div>
            </div>
            
            {/* Technique summary */}
            {techniques.length > 0 && (
              <div className="rounded-lg border bg-card p-4">
                <h4 className="text-sm font-semibold mb-2">Available Jailbreak Techniques ({techniques.length} types)</h4>
                <div className="flex flex-wrap gap-2">
                  {techniques.map((tech) => {
                    const count = techniqueCounts[tech] || 0
                    return (
                      <button
                        key={tech}
                        onClick={() => setTechniqueFilter(tech === techniqueFilter ? '' : tech)}
                        className={`text-xs px-3 py-1 rounded border transition-colors ${
                          techniqueFilter === tech
                            ? 'bg-primary text-primary-foreground border-primary'
                            : 'bg-muted hover:bg-muted/80 border-border'
                        }`}
                      >
                        {tech.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} ({count})
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Info for benchmark prompts */}
        {showBenchmarks && (categoryFilter === 'benchmark' || categoryFilter === '') && (
          <div className="rounded-lg border bg-purple-50 dark:bg-purple-950/20 border-purple-200 dark:border-purple-800 p-4">
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <h3 className="font-semibold text-sm mb-1">Benchmark Prompts</h3>
                <p className="text-xs text-muted-foreground">
                  These are standardized benchmark test prompts from various LLM evaluation datasets.
                  Prompts are randomized and deduplicated for uniqueness.
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="text-sm text-muted-foreground">
          {isLoading ? (
            'Loading prompts...'
          ) : (
            <>
              Showing {sortedPrompts.length} of {visibleCount} visible prompts
              {showJailbreaks && jailbreakCount > 0 && ` (${jailbreakCount} jailbreak prompts shown)`}
              {showBenchmarks && benchmarkCount > 0 && ` (${benchmarkCount} benchmark prompts shown)`}
              {categoryFilter === 'adversarial' && techniqueFilter && ` • Filtered by technique: ${techniqueFilter.replace(/_/g, ' ')}`}
              {totalLoaded === 0 && ' • Try changing the category filter or check if prompts are loaded'}
            </>
          )}
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {sortedPrompts.map((prompt) => (
            <PromptCard key={prompt.id} prompt={prompt} onDelete={handleDelete} />
          ))}
        </div>

        {!isLoading && sortedPrompts.length === 0 && (
          <div className="rounded-lg border bg-card p-8 text-center">
            {error ? (
              <div>
                <p className="text-destructive mb-2">Error loading prompts: {String(error)}</p>
                <p className="text-sm text-muted-foreground">Please check the API connection and try again.</p>
              </div>
            ) : categoryFilter === 'adversarial' && techniqueFilter ? (
              <div>
                <p className="text-muted-foreground mb-2">No jailbreak prompts found for technique: <strong>{techniqueFilter}</strong></p>
                <button
                  onClick={() => setTechniqueFilter('')}
                  className="text-sm text-primary hover:underline"
                >
                  Clear technique filter
                </button>
              </div>
            ) : categoryFilter ? (
              <div>
                <p className="text-muted-foreground mb-2">No prompts found in category: <strong>{categoryFilter}</strong></p>
                <button
                  onClick={() => setCategoryFilter('')}
                  className="text-sm text-primary hover:underline"
                >
                  Show all prompts
                </button>
              </div>
            ) : (
              <div>
                <p className="text-muted-foreground mb-2">No prompts found.</p>
                <p className="text-sm text-muted-foreground">
                  {totalLoaded === 0 
                    ? 'No prompts are loaded. Check if the API is running and prompts are imported.'
                    : 'Try adjusting your search or filters.'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  )
}

function PromptCard({ prompt, onDelete }: { prompt: Prompt; onDelete: (id: number) => void }) {
  const isTestPrompt = prompt.prompt_type === 'test_prompt'
  
  return (
    <div className="rounded-lg border bg-card p-6 space-y-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="font-semibold text-lg mb-1">{getPromptDisplayName(prompt)}</h3>
          {prompt.description && <p className="text-sm text-muted-foreground line-clamp-2">{prompt.description}</p>}
        </div>
        <div className="flex gap-2">
          {isTestPrompt && (
            <Link
              href={`/create-test?promptId=${prompt.id}`}
              className="p-2 hover:bg-muted rounded text-primary"
              title="Run Test with this Prompt"
            >
              <Play className="h-4 w-4" />
            </Link>
          )}
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

      {/* Show technique prominently for jailbreak prompts */}
      {prompt.category === 'adversarial' && prompt.metadata?.jailbreak_technique && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-primary">Technique:</span>
          <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded font-medium">
            {prompt.metadata.jailbreak_technique.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </span>
        </div>
      )}
      
      {prompt.tags && prompt.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {prompt.tags
            .filter(tag => tag !== 'jailbreak' && tag !== 'adversarial' && tag !== 'imported') // Hide redundant tags
            .slice(0, 5) // Limit to 5 tags
            .map((tag, idx) => (
              <span key={idx} className="text-xs bg-muted px-2 py-1 rounded">
                {tag}
              </span>
            ))}
        </div>
      )}
      
      {/* Run Test button for test prompts */}
      {isTestPrompt && (
        <div className="pt-2 border-t">
          <Link
            href={`/create-test?promptId=${prompt.id}`}
            className="flex items-center justify-center gap-2 w-full px-4 py-2 text-sm font-medium text-primary-foreground bg-primary hover:bg-primary/90 rounded-lg transition-colors"
          >
            <Play className="h-4 w-4" />
            Run Test with this Prompt
          </Link>
        </div>
      )}
    </div>
  )
}
