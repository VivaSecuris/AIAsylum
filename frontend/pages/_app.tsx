import type { AppProps } from 'next/app'
import Head from 'next/head'
import { useEffect } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { ToastContainer } from '@/lib/toast'
import { AuthProvider } from '@/lib/auth'
import axios from 'axios'
import '../styles/globals.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: (failureCount, error: any) => {
        // Authentication requires a session change, not another identical request.
        if (error?.response?.status === 401 || error?.response?.status === 403) return false
        // Retry network errors more aggressively
        if (error?.code === 'ECONNREFUSED' || error?.message?.includes('Network Error') || error?.code === 'ERR_NETWORK') {
          return failureCount < 3 // Retry up to 3 times for network errors
        }
        return failureCount < 1 // Default retry once
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000), // Exponential backoff
      staleTime: 5000,
    },
  },
})

export default function App({ Component, pageProps }: AppProps) {
  // Check API server health on app startup
  useEffect(() => {
    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    axios.get(`${API_BASE_URL}/health`, { timeout: 5000 })
      .then(() => {
        console.log('[App] API server is healthy')
      })
      .catch((error) => {
        console.error('[App] API server health check failed:', error)
        console.error('[App] Make sure the backend is running at', API_BASE_URL)
      })
  }, [])
  return (
    <>
      <Head>
        <title>LLM Psychoanalysis Framework</title>
        <meta name="description" content="LLM Psychoanalysis Framework - AI Safety Testing Framework" />
      </Head>
      <ErrorBoundary>
        <QueryClientProvider client={queryClient}>
          <AuthProvider>
            <Component {...pageProps} />
            <ToastContainer />
          </AuthProvider>
        </QueryClientProvider>
      </ErrorBoundary>
    </>
  )
}
