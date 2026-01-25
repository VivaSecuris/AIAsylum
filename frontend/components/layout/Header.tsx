import { useState } from 'react'
import { LogOut, Key } from 'lucide-react'
import { apiClient } from '@/lib/api'
import { useRouter } from 'next/router'

export function Header() {
  const [showLogin, setShowLogin] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const router = useRouter()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await apiClient.login(apiKey)
      setIsAuthenticated(true)
      setShowLogin(false)
      setApiKey('')
    } catch (error) {
      console.error('Login failed:', error)
      alert('Login failed. Please check your API key.')
    }
  }

  const handleLogout = async () => {
    try {
      await apiClient.logout()
      setIsAuthenticated(false)
    } catch (error) {
      console.error('Logout failed:', error)
    }
  }

  return (
    <header className="flex h-16 items-center justify-between border-b bg-card px-6">
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-semibold">LLM Psychoanalyst Framework</h2>
      </div>
      <div className="flex items-center gap-4">
        {!isAuthenticated ? (
          <>
            <button
              onClick={() => setShowLogin(true)}
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
                      placeholder="API Key"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="rounded border px-3 py-2 text-sm"
                      autoFocus
                    />
                    <div className="flex gap-2">
                      <button
                        type="submit"
                        className="flex-1 rounded bg-primary px-3 py-2 text-sm text-primary-foreground hover:bg-primary/90"
                      >
                        Login
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
