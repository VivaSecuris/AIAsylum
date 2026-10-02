import { ReactNode } from 'react'
import { KeyRound } from 'lucide-react'

import { useAuthSession } from '@/lib/auth'
import { Sidebar } from './Sidebar'
import { Header } from './Header'
import { LoginForm } from './LoginForm'

interface LayoutProps {
  children: ReactNode
}

/** Shown instead of the page when the server requires a key and there is no session. */
function SignInScreen() {
  return (
    <div className="mx-auto mt-16 max-w-md space-y-4 rounded-lg border bg-card p-6">
      <div className="flex items-center gap-2">
        <KeyRound className="h-5 w-5" />
        <h1 className="text-xl font-semibold">Sign in</h1>
      </div>
      <p className="text-sm text-muted-foreground">
        This server requires an API key. Use one of the keys in the server&apos;s API_KEYS setting; the session lasts until you sign out or it expires.
      </p>
      <LoginForm />
    </div>
  )
}

export function Layout({ children }: LayoutProps) {
  const { authRequired, isAuthenticated, isCheckingSession } = useAuthSession()
  const signedOut = authRequired && !isAuthenticated && !isCheckingSession
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto bg-background p-6">
          {signedOut ? <SignInScreen /> : children}
        </main>
      </div>
    </div>
  )
}
