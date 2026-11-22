import { useCallback, useEffect, useState } from 'react'

import { api } from './api'

export type ConfigYamlStatus = 'error' | 'idle' | 'loading' | 'ready'

export function useConfigYamlSource(profile: string, enabled: boolean) {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<ConfigYamlStatus>('idle')
  const [retryKey, setRetryKey] = useState(0)

  const retry = useCallback(() => {
    setText('')
    setLoading(true)
    setStatus('loading')
    setRetryKey(current => current + 1)
  }, [])

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    setLoading(true)
    setStatus('loading')

    api.getConfigRaw(profile || undefined)
      .then(response => {
        if (!cancelled) {
          setText(response.yaml)
          setStatus('ready')
        }
      })
      .catch(() => {
        if (!cancelled) {
          setText('')
          setStatus('error')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [enabled, profile, retryKey])

  return { loading, retry, setText, status, text }
}
