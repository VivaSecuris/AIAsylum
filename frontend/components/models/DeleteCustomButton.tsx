import { useState } from 'react'
import { useDeleteCustomModels } from '@/lib/model-catalog'
import { formatApiError } from '@/lib/utils'

export function DeleteCustomButton({ names }: { names: string[] }) {
  const [confirm, setConfirm] = useState(false)
  const remove = useDeleteCustomModels()
  if (!names.length) return null
  return <div className="space-y-2 text-xs">
    {!confirm ? <button type="button" onClick={() => setConfirm(true)} className="text-destructive hover:underline">{names.length === 1 ? 'Delete custom checkpoint' : `Delete all ${names.length} custom checkpoints`}</button> : <div className="space-y-2 rounded-md border border-destructive/30 p-3">
      <p>Delete {names.length === 1 ? names[0] : `${names.length} custom checkpoints`} from this server? Their weights will be permanently removed. Saved results and history stay available.</p>
      <div className="flex gap-3"><button type="button" disabled={remove.isPending} onClick={() => remove.mutate(names, { onSuccess: () => setConfirm(false) })} className="rounded bg-destructive px-3 py-2 text-destructive-foreground disabled:opacity-50">{remove.isPending ? 'Deleting…' : 'Delete weights'}</button><button type="button" disabled={remove.isPending} onClick={() => setConfirm(false)}>Keep checkpoints</button></div>
    </div>}
    {remove.error && <p role="alert" className="text-destructive">{formatApiError(remove.error, 'Could not delete the custom checkpoints.')}</p>}
  </div>
}
