import { useId, useState } from 'react'
import { FolderPlus, Pencil } from 'lucide-react'
import { UNASSIGNED_EXPERIMENT, useModelOrganization, useSaveExperiment } from '@/lib/model-organization'
import { formatApiError } from '@/lib/utils'

const BUTTON = 'inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50'

export function ExperimentToolbar({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const { data, isLoading, error, refetch } = useModelOrganization()
  const selected = data?.experiments.find((experiment) => experiment.id === value)
  const [editing, setEditing] = useState<'new' | string | null>(null)
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [createdName, setCreatedName] = useState('')
  const save = useSaveExperiment()
  const id = useId()

  async function submit() {
    try {
      const saved = await save.mutateAsync({ id: editing === 'new' ? undefined : editing!, name: name.trim(), notes: notes.trim() })
      const creating = editing === 'new'
      setEditing(null)
      if (creating) {
        setCreatedName(saved.name)
        onChange('')
      } else if (saved?.id) onChange(saved.id)
    } catch { /* The mutation error is displayed in the form. */ }
  }

  return <section aria-label="Organize experiments" className="space-y-3 rounded-xl border bg-card p-4">
    <div className="flex flex-wrap items-end gap-3">
      <div className="min-w-48 flex-1"><label htmlFor={`${id}-filter`} className="text-sm font-medium">Experiment</label><select id={`${id}-filter`} disabled={isLoading || !!error} value={value} onChange={(event) => { onChange(event.target.value); setEditing(null); setCreatedName(''); save.reset() }} className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"><option value="">All experiments</option><option value={UNASSIGNED_EXPERIMENT}>Unassigned models and steps</option>{data?.experiments.map((experiment) => <option key={experiment.id} value={experiment.id}>{experiment.name}</option>)}</select></div>
      <button type="button" disabled={isLoading || !!error} onClick={() => { setEditing('new'); setName(''); setNotes(''); save.reset() }} className={BUTTON}><FolderPlus className="h-4 w-4" /> New experiment</button>
      {selected && <button type="button" onClick={() => { setEditing(selected.id); setName(selected.name); setNotes(selected.notes); save.reset() }} className={BUTTON}><Pencil className="h-4 w-4" /> Edit experiment</button>}
    </div>
    {createdName && <p role="status" className="rounded-md bg-emerald-50 p-3 text-sm text-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-200">Created “{createdName}”. Open Organize on a model or selected history step, check this experiment, and save.</p>}
    {error ? <div role="alert" className="flex flex-wrap items-center gap-3 text-sm text-destructive"><span>{formatApiError(error, 'Could not load experiments.')}</span><button type="button" onClick={() => refetch()} className={BUTTON}>Try again</button></div> : <p className="text-xs text-muted-foreground">{selected?.notes || 'Group related models and steps into a named experiment. Use Organize on a model or history step to add it.'}</p>}
    {editing && <div className="space-y-3 border-t pt-3">
      <h2 className="text-sm font-semibold">{editing === 'new' ? 'New experiment' : 'Edit experiment'}</h2>
      <div className="grid gap-3 sm:grid-cols-2"><div><label htmlFor={`${id}-name`} className="text-xs font-medium">Name</label><input id={`${id}-name`} autoFocus value={name} onChange={(event) => setName(event.target.value)} maxLength={120} placeholder="e.g. Qwen refusal direction study" className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm" /></div><div><label htmlFor={`${id}-notes`} className="text-xs font-medium">Purpose or notes</label><input id={`${id}-notes`} value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={4000} placeholder="What are these runs investigating?" className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm" /></div></div>
      {save.error && <p role="alert" className="text-sm text-destructive">{formatApiError(save.error, 'Could not save this experiment.')}</p>}
      <div className="flex gap-2"><button type="button" disabled={!name.trim() || save.isPending} onClick={submit} className={`${BUTTON} border-primary bg-primary text-primary-foreground hover:bg-primary/90`}>{save.isPending ? 'Saving…' : 'Save experiment'}</button><button type="button" disabled={save.isPending} onClick={() => setEditing(null)} className={BUTTON}>Cancel</button></div>
    </div>}
  </section>
}
