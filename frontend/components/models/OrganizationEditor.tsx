import { useEffect, useId, useState } from 'react'
import { useModelOrganization, useSaveOrganizationItem } from '@/lib/model-organization'
import { formatApiError } from '@/lib/utils'

const FIELD = 'mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm'

export function OrganizationEditor({ itemKey, originalLabel }: { itemKey: string; originalLabel: string }) {
  const { data, isLoading, error } = useModelOrganization()
  const item = data?.items[itemKey]
  const [label, setLabel] = useState(item?.label ?? '')
  const [notes, setNotes] = useState(item?.notes ?? '')
  const [experimentIds, setExperimentIds] = useState(item?.experiment_ids ?? [])
  const save = useSaveOrganizationItem()
  const id = useId()
  useEffect(() => {
    setLabel(item?.label ?? '')
    setNotes(item?.notes ?? '')
    setExperimentIds(item?.experiment_ids ?? [])
  }, [itemKey, item])
  useEffect(() => { save.reset() }, [itemKey])

  return <details className="rounded-lg border bg-background/50 p-3">
    <summary className="cursor-pointer text-sm font-medium">Organize · name, notes and experiments</summary>
    {error ? <p role="alert" className="mt-3 text-sm text-destructive">{formatApiError(error, 'Could not load organization settings.')}</p> : isLoading ? <p className="mt-3 text-sm text-muted-foreground">Loading organization settings…</p> : <div className="mt-3 space-y-3">
      <div><label htmlFor={`${id}-name`} className="text-xs font-medium">Display name</label><input id={`${id}-name`} value={label} maxLength={120} onChange={(event) => { setLabel(event.target.value); save.reset() }} placeholder={originalLabel} className={FIELD} /></div>
      <div><label htmlFor={`${id}-notes`} className="text-xs font-medium">Notes</label><textarea id={`${id}-notes`} value={notes} maxLength={4000} rows={3} onChange={(event) => { setNotes(event.target.value); save.reset() }} placeholder="What are you trying, and what should happen?" className={FIELD} /></div>
      <fieldset><legend className="text-xs font-medium">Experiments</legend>{data?.experiments.length ? <div className="mt-2 max-h-36 space-y-2 overflow-auto">{data.experiments.map((experiment) => <label key={experiment.id} className="flex items-start gap-2 text-sm"><input type="checkbox" checked={experimentIds.includes(experiment.id)} onChange={(event) => { setExperimentIds((ids) => event.target.checked ? [...ids, experiment.id] : ids.filter((value) => value !== experiment.id)); save.reset() }} className="mt-0.5" /><span className="break-words">{experiment.name}</span></label>)}</div> : <p className="mt-1 text-xs text-muted-foreground">Use New experiment above to create a group for related models and steps.</p>}</fieldset>
      <p className="text-xs text-muted-foreground">Names and notes are saved on this server. Model files and recorded history keep their original identities.</p>
      {save.error && <p role="alert" className="text-sm text-destructive">{formatApiError(save.error, 'Could not save organization settings.')}</p>}
      <div className="flex items-center gap-3"><button type="button" disabled={save.isPending} onClick={() => save.mutate({ key: itemKey, label: label.trim(), notes: notes.trim(), experiment_ids: experimentIds })} className="rounded-md border border-primary bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">{save.isPending ? 'Saving…' : 'Save organization'}</button>{save.isSuccess && <span role="status" className="text-xs text-emerald-700 dark:text-emerald-400">Saved</span>}</div>
    </div>}
  </details>
}
