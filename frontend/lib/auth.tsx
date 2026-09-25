import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { apiClient } from './api'

interface AuthSession {
  isAuthenticated: boolean
  isUpdating: boolean
  isCheckingSession: boolean
  login: (apiKey: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthSession | null>(null)

/** Session UI survives page navigation; the actual credential is an HttpOnly cookie. */
export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient()
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isUpdating, setIsUpdating] = useState(false)
  const [isCheckingSession, setIsCheckingSession] = useState(true)
  const sessionVersion = useRef(0)

  useEffect(() => {
    let cancelled = false
    const version = sessionVersion.current
    apiClient.getSession()
      .then((session) => {
        if (!cancelled && version === sessionVersion.current) {
          setIsAuthenticated(session.authenticated)
        }
      })
      .catch(() => {
        // A disconnected server should still leave the login form accessible.
      })
      .finally(() => {
        if (!cancelled && version === sessionVersion.current) setIsCheckingSession(false)
      })
    return () => { cancelled = true }
  }, [])

  async function updateSession(authenticated: boolean, apiKey?: string) {
    // A delayed startup response must never overwrite an explicit session change.
    sessionVersion.current += 1
    setIsCheckingSession(false)
    setIsUpdating(true)
    try {
      if (authenticated) await apiClient.login(apiKey ?? '')
      else await apiClient.logout()
      setIsAuthenticated(authenticated)
      // Ignore requests started with the old cookie. Reset removes cached 401s
      // on login and protected data on logout, then refetches active queries.
      await queryClient.cancelQueries()
      await queryClient.resetQueries()
    } finally {
      setIsUpdating(false)
    }
  }

  return (
    <AuthContext.Provider value={{
      isAuthenticated,
      isUpdating,
      isCheckingSession,
      login: (apiKey) => updateSession(true, apiKey),
      logout: () => updateSession(false),
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuthSession() {
  const session = useContext(AuthContext)
  if (!session) throw new Error('useAuthSession must be used inside AuthProvider')
  return session
}
