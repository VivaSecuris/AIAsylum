import { useState } from 'react'
import { LogOut, Key } from 'lucide-react'
import { useAuthSession } from '@/lib/auth'
import { formatApiError } from '@/lib/utils'
import { toast } from '@/lib/toast'

export function Header() {
  const [showLogin, setShowLogin] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const { isAuthenticated, isUpdating, isCheckingSession, login, logout } = useAuthSession()
  const [loginError, setLoginError] = useState<string | null>(null)

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    if (isUpdating) return
    setLoginError(null)
    try {
      await login(apiKey)
      setShowLogin(false)
      setApiKey('')
      toast.success('Signed in.')
    } catch (error) {
      setLoginError(formatApiError(error, 'Login failed. Please check your API key.'))
    }
  }

  const handleLogout = async () => {
    try {
      await logout()
      toast.success('Signed out.')
    } catch (error) {
      toast.error(formatApiError(error, 'Could not sign out'))
    }
  }

  return (
    <header className="flex h-16 items-center justify-between border-b bg-card px-6">
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-semibold">LLM Psychoanalysis Framework</h2>
      </div>
      <div className="flex items-center gap-4">
        {isUpdating && <span role="status" className="text-xs text-muted-foreground">Refreshing server status…</span>}
        {isCheckingSession ? (
          <span role="status" className="text-xs text-muted-foreground">Checking session…</span>
        ) : !isAuthenticated ? (
          <>
            <button
              onClick={() => { setShowLogin(true); setLoginError(null) }}
              disabled={isUpdating}
              className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              <Key className="h-4 w-4" />
              Login
            </button>
            {showLogin && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  onClick={() => {
                    setShowLogin(false)
                    setApiKey('')
                  }}
                />
                <div className="absolute right-6 top-16 z-50 rounded-lg border bg-card p-4 shadow-lg min-w-[300px]">
                  <form onSubmit={handleLogin} className="flex flex-col gap-3">
                    <input
                      type="password"
                      aria-label="API key"
                      placeholder="API Key"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      autoFocus
                    />
                    {loginError && <p role="alert" className="max-w-sm text-xs text-destructive">{loginError}</p>}
                    <div className="flex gap-2">
                      <button
                        type="submit"
                        disabled={isUpdating || !apiKey.trim()}
                        className="flex-1 rounded bg-primary px-3 py-2 text-sm text-primary-foreground hover:bg-primary/90"
                      >
                        {isUpdating ? 'Signing in…' : 'Login'}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setShowLogin(false)
                          setApiKey('')
                        }}
                        className="rounded border px-3 py-2 text-sm hover:bg-muted"
                      >
                        Cancel
                      </button>
                    </div>
                  </form>
                </div>
              </>
            )}
          </>
        ) : (
          <button
            onClick={handleLogout}
            disabled={isUpdating}
            className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
          >
            <LogOut className="h-4 w-4" />
            Logout
          </button>
        )}
      </div>
    </header>
  )
}
