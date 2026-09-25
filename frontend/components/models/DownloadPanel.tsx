import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Download, HardDrive, Trash2, X } from 'lucide-react'

import {
  useCancelModelDownload,
  useDeleteCachedModel,
  useModelDownloads,
  useStartModelDownload,
  type ModelDownload,
} from '@/lib/model-catalog'
import { formatApiError } from '@/lib/utils'

const ACTION = 'inline-flex items-center justify-center gap-2 rounded-md border px-3 py-2 text-sm font-medium transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50'
const INPUT = 'rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

export function readableBytes(bytes: number | null | undefined) {
  if (bytes == null || bytes < 0) return '—'
  const gib = bytes / 1024 ** 3
  if (gib >= 1) return `${gib.toFixed(1)} GiB`
  const mib = bytes / 1024 ** 2
  return mib >= 1 ? `${mib.toFixed(0)} MiB` : `${(bytes / 1024).toFixed(0)} KiB`
}

function statusLabel(d: ModelDownload) {
  switch (d.status) {
    case 'queued': return 'Queued'
    case 'resolving': return 'Contacting Hugging Face…'
    case 'downloading': return 'Downloading'
    case 'completed': return d.already_present ? 'Already on server' : 'Downloaded'
    case 'failed': return 'Failed'
    case 'cancelled': return 'Cancelled'
  }
}

export function DownloadProgress({ download, onCancel, cancelling }: { download: ModelDownload; onCancel?: () => void; cancelling?: boolean }) {
  const total = download.bytes_total ?? 0
  const done = Math.min(download.bytes_done, total || download.bytes_done)
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : null
  const tone = download.status === 'failed' ? 'text-destructive' : download.status === 'completed' ? 'text-emerald-700 dark:text-emerald-400' : 'text-muted-foreground'
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className={`font-medium ${tone}`}>
          {statusLabel(download)}
          {download.active && pct != null && ` · ${pct}%`}
          {download.status === 'downloading' && total > 0 && ` · ${readableBytes(done)} of ${readableBytes(total)}`}
          {download.status === 'completed' && !download.already_present && ` · ${readableBytes(download.bytes_done)} in ${Math.round(download.elapsed_seconds)}s`}
        </span>
        {download.active && onCancel && (
          <button type="button" onClick={onCancel} disabled={cancelling} className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground disabled:opacity-50">
            <X className="h-3 w-3" /> Cancel
          </button>
        )}
      </div>
      {download.active && (
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted" role="progressbar" aria-valuenow={pct ?? undefined} aria-valuemin={0} aria-valuemax={100}>
          <div className={`h-full rounded-full bg-primary ${pct == null ? 'w-1/4 animate-pulse' : ''}`} style={pct != null ? { width: `${pct}%` } : undefined} />
        </div>
      )}
      {download.error && <p className="text-xs text-destructive">{download.error}</p>}
    </div>
  )
}

/** Start a download for one model. Shows its progress once started. */
export function DownloadModelButton({ repoId, download, className }: { repoId: string; download?: ModelDownload; className?: string }) {
  const start = useStartModelDownload()
  const cancel = useCancelModelDownload()
  if (download && (download.active || download.status === 'failed' || download.status === 'cancelled')) {
    return (
      <div className="space-y-2">
        <DownloadProgress download={download} onCancel={() => cancel.mutate(download.id)} cancelling={cancel.isPending} />
        {!download.active && (
          <button type="button" onClick={() => start.mutate({ repo_id: repoId })} disabled={start.isPending} className={className ?? ACTION}>
            <Download className="h-4 w-4" /> Try again
          </button>
        )}
      </div>
    )
  }
  return (
    <div className="space-y-1">
      <button type="button" onClick={() => start.mutate({ repo_id: repoId })} disabled={start.isPending} className={className ?? ACTION}>
        <Download className="h-4 w-4" /> {start.isPending ? 'Starting…' : 'Download to server'}
      </button>
      {start.error && <p className="text-xs text-destructive">{formatApiError(start.error, 'Could not start the download')}</p>}
    </div>
  )
}

/** Remove a cached base model. The confirmation is inline: the viewer shows no dialogs. */
export function RemoveCachedButton({ repoId, sizeBytes }: { repoId: string; sizeBytes?: number | null }) {
  const [confirming, setConfirming] = useState(false)
  const remove = useDeleteCachedModel()
  if (confirming) {
    return (
      <div className="space-y-2 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-xs">
        <p>Delete {readableBytes(sizeBytes)} of cached weights for <span className="font-mono">{repoId}</span> from this server? Runs that use it will download it again.</p>
        <div className="flex gap-2">
          <button type="button" onClick={() => remove.mutate(repoId, { onSettled: () => setConfirming(false) })} disabled={remove.isPending} className={`${ACTION} border-destructive text-destructive hover:bg-destructive/10`}>
            <Trash2 className="h-3.5 w-3.5" /> {remove.isPending ? 'Deleting…' : 'Delete from server'}
          </button>
          <button type="button" onClick={() => setConfirming(false)} className={ACTION}>Keep</button>
        </div>
        {remove.error && <p className="text-destructive">{formatApiError(remove.error, 'Could not delete')}</p>}
      </div>
    )
  }
  return (
    <button type="button" onClick={() => setConfirming(true)} className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-destructive">
      <Trash2 className="h-3 w-3" /> Remove from server
    </button>
  )
}

/** The pull form and the list of downloads on the connected server. */
export function DownloadPanel({ location }: { location?: string }) {
  const { data, error } = useModelDownloads()
  const start = useStartModelDownload()
  const cancel = useCancelModelDownload()
  const queryClient = useQueryClient()
  const [repoId, setRepoId] = useState('')
  const [revision, setRevision] = useState('')
  const downloads = data?.downloads ?? []
  const activeCount = downloads.filter((d) => d.active).length

  // When a download finishes the catalog changes; refresh it once, not on every poll.
  const lastActive = useRef(activeCount)
  useEffect(() => {
    if (activeCount < lastActive.current) queryClient.invalidateQueries({ queryKey: ['model-catalog'] })
    lastActive.current = activeCount
  }, [activeCount, queryClient])

  const submit = (event: React.FormEvent) => {
    event.preventDefault()
    const id = repoId.trim()
    if (!id) return
    start.mutate({ repo_id: id, revision: revision.trim() || undefined }, { onSuccess: () => { setRepoId(''); setRevision('') } })
  }

  return (
    <section className="space-y-4 rounded-xl border bg-card p-5" aria-labelledby="downloads-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="downloads-heading" className="flex items-center gap-2 text-base font-semibold"><Download className="h-4 w-4" /> Download a model to {location || 'the server'}</h2>
          <p className="mt-1 text-sm text-muted-foreground">Fetches a Hugging Face checkpoint into the server&apos;s cache so runs start without a download. Only config, tokenizer and safetensors weights are fetched.</p>
        </div>
        {data && (
          <p className="inline-flex items-center gap-1.5 text-xs text-muted-foreground" title={data.cache_dir}>
            <HardDrive className="h-3.5 w-3.5" /> {readableBytes(data.free_bytes)} free of {readableBytes(data.total_bytes)}
          </p>
        )}
      </div>

      <form onSubmit={submit} className="flex flex-wrap items-end gap-2">
        <div className="min-w-[220px] flex-1">
          <label htmlFor="download-repo" className="text-xs text-muted-foreground">Hugging Face id</label>
          <input id="download-repo" value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="Qwen/Qwen2.5-3B-Instruct" className={`mt-1 w-full ${INPUT}`} autoComplete="off" spellCheck={false} />
        </div>
        <div className="w-36">
          <label htmlFor="download-revision" className="text-xs text-muted-foreground">Revision</label>
          <input id="download-revision" value={revision} onChange={(e) => setRevision(e.target.value)} placeholder="main" className={`mt-1 w-full ${INPUT}`} autoComplete="off" spellCheck={false} />
        </div>
        <button type="submit" disabled={!repoId.trim() || start.isPending} className={`${ACTION} border-primary bg-primary text-primary-foreground hover:bg-primary/90`}>
          <Download className="h-4 w-4" /> {start.isPending ? 'Starting…' : 'Download'}
        </button>
      </form>
      {start.error && <p role="alert" className="text-xs text-destructive">{formatApiError(start.error, 'Could not start the download')}</p>}
      <p className="text-xs text-muted-foreground">Gated models need the server to hold a Hugging Face token (<code>HF_TOKEN</code> or <code>huggingface-cli login</code> on that machine). Cancelling keeps what was fetched; the next download resumes from it.</p>

      {error && <p role="alert" className="text-xs text-destructive">{formatApiError(error, 'Could not list downloads')}</p>}
      {downloads.length > 0 && (
        <ul className="divide-y rounded-md border">
          {downloads.map((d) => (
            <li key={d.id} className="space-y-2 p-3">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-mono text-sm">{d.repo_id}{d.revision !== 'main' ? `@${d.revision}` : ''}</span>
                {d.files_total != null && d.bytes_total != null && d.status !== 'completed' && <span className="text-xs text-muted-foreground">{d.files_total} files · {readableBytes(d.bytes_total)}</span>}
              </div>
              <DownloadProgress download={d} onCancel={() => cancel.mutate(d.id)} cancelling={cancel.isPending} />
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
