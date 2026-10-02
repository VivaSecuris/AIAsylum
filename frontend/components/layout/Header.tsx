import { LogOut, ShieldCheck, ShieldOff } from 'lucide-react'
import { useAuthSession } from '@/lib/auth'
import { formatApiError } from '@/lib/utils'
import { toast } from '@/lib/toast'

/**
 * Session status. A Login control only appears when the server requires an API
 * key (REQUIRE_AUTH); without it nothing is locked, so the header says so
 * instead of offering a login that would change nothing. Signing in itself
 * happens on the signed-out screen in Layout.
 */
export function Header() {
  const { authRequired, isAuthenticated, isUpdating, isCheckingSession, logout } = useAuthSession()

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
        ) : !authRequired ? (
          <span
            className="flex items-center gap-1.5 text-xs text-muted-foreground"
            title="This server does not require an API key (REQUIRE_AUTH is off), so anyone who can reach it can use it. Keep it on 127.0.0.1, or set REQUIRE_AUTH=true to require a key."
          >
            <ShieldOff className="h-3.5 w-3.5" /> No sign-in required
          </span>
        ) : isAuthenticated ? (
          <>
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <ShieldCheck className="h-3.5 w-3.5" /> Signed in
            </span>
            <button
              onClick={handleLogout}
              disabled={isUpdating}
              className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              <LogOut className="h-4 w-4" />
              Sign out
            </button>
          </>
        ) : (
          <span className="text-xs text-muted-foreground">Signed out</span>
        )}
      </div>
    </header>
  )
}
