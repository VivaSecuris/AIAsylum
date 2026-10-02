import { useState } from 'react'

import { useAuthSession } from '@/lib/auth'
import { toast } from '@/lib/toast'
import { formatApiError } from '@/lib/utils'

/** API-key sign-in, used by the signed-out screen. */
export function LoginForm({ onDone }: { onDone?: () => void }) {
  const { isUpdating, login } = useAuthSession()
  const [apiKey, setApiKey] = useState('')
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (isUpdating) return
    setError(null)
    try {
      await login(apiKey)
      setApiKey('')
      toast.success('Signed in.')
      onDone?.()
    } catch (err) {
      setError(formatApiError(err, 'Sign-in failed. Check the API key.'))
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-3">
      <label className="text-sm font-medium">
        API key
        <input
          type="password"
          autoComplete="current-password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          autoFocus
        />
      </label>
      {error && <p role="alert" className="text-xs text-destructive">{error}</p>}
      <button
        type="submit"
        disabled={isUpdating || !apiKey.trim()}
        className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
      >
        {isUpdating ? 'Signing in…' : 'Sign in'}
      </button>
    </form>
  )
}
