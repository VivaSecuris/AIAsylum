import { useState } from 'react'
import Link from 'next/link'
import { ArrowDown, ArrowUp, X } from 'lucide-react'

import { useDebouncedValue, usePrompts, usePromptsByIds } from '@/lib/hooks'
import { isPatientTestPrompt } from '@/lib/prompt-targets'
import { getPromptDisplayName } from '@/lib/utils'
import { INPUT, SMALL_BUTTON } from './ui'

const RESULT_LIMIT = 50

/**
 * Search the test-prompt library and pick one prompt (one-shot) or an ordered
 * list (multi-shot). The search runs on the server, so every prompt is
 * reachable, not only the newest page; selections are resolved by ID.
 */
export function PromptPicker({ mode, selectedIds, onChange }: {
  mode: 'single' | 'multi'
  selectedIds: number[]
  onChange: (ids: number[]) => void
}) {
  const [search, setSearch] = useState('')
  const debounced = useDebouncedValue(search.trim())
  const { data: allResults = [], isFetching } = usePrompts({
    prompt_type: 'test_prompt',
    search: debounced || undefined,
    limit: RESULT_LIMIT,
  })
  const results = allResults.filter(isPatientTestPrompt)
  const selected = usePromptsByIds(selectedIds)
  const hasInvalidSelection = selected.some((query) => query.data && !isPatientTestPrompt(query.data))

  const choose = (id: number) => {
    if (!results.some((prompt) => prompt.id === id)) return
    if (mode === 'single') onChange([id])
    else if (!selectedIds.includes(id)) onChange([...selectedIds, id])
  }
  const move = (index: number, offset: number) => {
    const ordered = [...selectedIds]
    ;[ordered[index], ordered[index + offset]] = [ordered[index + offset], ordered[index]]
    onChange(ordered)
  }

  return (
    <div className="space-y-3">
      {selectedIds.length > 0 && (
        <ol className="space-y-2">
          {selectedIds.map((id, index) => {
            const query = selected[index]
            const prompt = query?.data
            const invalid = prompt && !isPatientTestPrompt(prompt)
            return (
              <li key={id} className="rounded-md border bg-primary/5 p-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0 flex-1 text-sm">
                    {mode === 'multi' && <span className="mr-2 text-muted-foreground">{index + 1}.</span>}
                    <span className="font-medium">
                      {prompt ? getPromptDisplayName(prompt) : query?.isError ? `Prompt #${id} (not found)` : `Prompt #${id}`}
                    </span>
                    {prompt?.category && <span className="ml-2 rounded bg-muted px-1.5 py-0.5 text-xs">{prompt.category}</span>}
                  </div>
                  <div className="flex gap-1">
                    {mode === 'multi' && (
                      <>
                        <button type="button" className={SMALL_BUTTON} disabled={index === 0} onClick={() => move(index, -1)} aria-label={`Move prompt ${index + 1} earlier`}><ArrowUp className="h-3.5 w-3.5" /></button>
                        <button type="button" className={SMALL_BUTTON} disabled={index === selectedIds.length - 1} onClick={() => move(index, 1)} aria-label={`Move prompt ${index + 1} later`}><ArrowDown className="h-3.5 w-3.5" /></button>
                      </>
                    )}
                    <button type="button" className={SMALL_BUTTON} onClick={() => onChange(selectedIds.filter((x) => x !== id))} aria-label={`Remove prompt ${index + 1}`}><X className="h-3.5 w-3.5" /></button>
                  </div>
                </div>
                {query?.isError && <p className="mt-1 text-xs text-destructive">This prompt no longer exists. Remove it or choose another.</p>}
                {invalid && <p role="alert" className="mt-1 text-xs text-destructive">This is not a patient test prompt. Remove it or choose a patient or general prompt.</p>}
                {prompt && !invalid && (
                  <details className="mt-1 text-xs text-muted-foreground">
                    <summary className="cursor-pointer">Show text</summary>
                    <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-3 font-sans">{prompt.prompt_text}</pre>
                  </details>
                )}
              </li>
            )
          })}
        </ol>
      )}
      {mode === 'multi' && selectedIds.length > 1 && (
        <p className="text-xs text-muted-foreground">The prompts run in this order in one conversation.</p>
      )}

      {(mode === 'multi' || selectedIds.length === 0 || hasInvalidSelection) && (
        <div className="space-y-2">
          <input
            type="search"
            aria-label="Search test prompts"
            placeholder="Search by name, text or #ID"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className={`${INPUT} mt-0`}
          />
          <div className="max-h-72 overflow-auto rounded-md border">
            {results.length === 0 ? (
              <p className="p-3 text-xs text-muted-foreground">
                {isFetching ? 'Searching…' : <>No prompts match. <Link href="/prompts/create" className="text-primary underline">Create a prompt</Link> or write a custom one.</>}
              </p>
            ) : (
              <ul className="divide-y">
                {results.map((prompt) => {
                  const picked = selectedIds.includes(prompt.id)
                  return (
                    <li key={prompt.id}>
                      <button
                        type="button"
                        disabled={picked}
                        onClick={() => choose(prompt.id)}
                        className="w-full px-3 py-2 text-left text-sm hover:bg-muted disabled:opacity-50"
                      >
                        <span className="font-medium">{getPromptDisplayName(prompt)}</span>
                        {prompt.category && <span className="ml-2 rounded bg-muted px-1.5 py-0.5 text-xs">{prompt.category}</span>}
                        <span className="mt-0.5 block truncate text-xs text-muted-foreground">{prompt.prompt_text}</span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            {allResults.length === RESULT_LIMIT ? `Showing the ${RESULT_LIMIT} newest matches; refine the search to see others. ` : ''}
            {mode === 'multi' ? 'Click prompts to add them in order.' : 'Click a prompt to use it.'}{' '}
            <Link href="/prompts" className="text-primary underline">Manage prompts</Link>
          </p>
        </div>
      )}
    </div>
  )
}
