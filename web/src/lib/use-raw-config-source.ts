import { useCallback, useEffect, useState } from 'react'

import { api } from './api'

export type RawConfigSourceStatus = 'error' | 'loading' | 'ready'

export function useRawConfigSource(profile: string) {
  const [source, setSource] = useState<Record<string, unknown> | null>(null)
  const [status, setStatus] = useState<RawConfigSourceStatus>('loading')
  const [retryKey, setRetryKey] = useState(0)

  const retry = useCallback(() => {
    setSource(null)
    setStatus('loading')
    setRetryKey(current => current + 1)
  }, [])

  useEffect(() => {
    let cancelled = false

    api.getConfigSource(profile || undefined)
      .then(next => {
        if (!cancelled) {
          setSource(next)
          setStatus('ready')
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSource(null)
          setStatus('error')
        }
      })

    return () => {
      cancelled = true
    }
  }, [profile, retryKey])

  return { retry, source, status }
}
