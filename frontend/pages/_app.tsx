import type { AppProps } from 'next/app'
import Head from 'next/head'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { ToastContainer } from '@/lib/toast'
import '../styles/globals.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5000,
    },
  },
})

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>LLM Psychoanalyst Framework</title>
        <meta name="description" content="LLM Psychoanalyst Framework - AI Safety Testing Framework" />
      </Head>
      <ErrorBoundary>
        <QueryClientProvider client={queryClient}>
          <Component {...pageProps} />
          <ToastContainer />
        </QueryClientProvider>
      </ErrorBoundary>
    </>
  )
}
