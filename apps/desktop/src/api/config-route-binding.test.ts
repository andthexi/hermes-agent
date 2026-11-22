import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getHermesConfigRecord,
  peekConfigReadOrigin,
  saveHermesConfig,
  setApiRequestConnection,
  setApiRequestProfile
} from '@/hermes'
import { withConfigDisplayLanguage } from '@/i18n'

describe('config read/write route binding', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn(async (request: { method?: string }) =>
      request.method === 'PUT' ? { ok: true } : { model: 'from-read' }
    )
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    setApiRequestConnection(null)
    setApiRequestProfile(null)
  })

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('keeps legacy normalized-raw query and profile routing beside an exact source view', async () => {
    setApiRequestConnection('connection-a')
    await getHermesConfigRecord('worker', { includeDefaults: false })
    await getHermesConfigRecord('worker', { rawSource: true })

    expect(api).toHaveBeenNthCalledWith(1, expect.objectContaining({
      connectionId: 'connection-a', profile: 'worker', path: '/api/config?include_defaults=false'
    }))
    expect(api).toHaveBeenNthCalledWith(2, expect.objectContaining({
      connectionId: 'connection-a', profile: 'worker', path: '/api/config?raw_source=true'
    }))
  })

  it('locale edits retain the captured route and only submit the explicit language path', async () => {
    setApiRequestConnection('connection-a')
    setApiRequestProfile('worker')
    const source = await getHermesConfigRecord(undefined, { includeDefaults: false })
    const localePatch = withConfigDisplayLanguage(source, 'fr')

    setApiRequestConnection('connection-b')
    await saveHermesConfig(localePatch, undefined, { preserveLanguage: true })

    expect(api.mock.calls.at(-1)?.[0]).toEqual(expect.objectContaining({
      connectionId: 'connection-a',
      path: '/api/config?preserve_language=true',
      profile: 'worker',
      method: 'PUT',
      body: {
        config: { display: { language: 'fr' } },
        write_mode: 'explicit_patch',
        unset_paths: []
      }
    }))
  })

  it('config record read from A cannot be written to B after primary changes', async () => {
    setApiRequestConnection('connection-a')
    setApiRequestProfile('default')

    const record = await getHermesConfigRecord()

    expect(peekConfigReadOrigin(record)).toEqual({ connectionId: 'connection-a', profile: 'default' })
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'connection-a', path: '/api/config', profile: 'default' })
    )

    setApiRequestConnection('connection-b')
    await saveHermesConfig(record)

    const puts = api.mock.calls.filter(call => call[0].method === 'PUT')

    expect(puts).toHaveLength(1)
    expect(puts[0][0]).toEqual(
      expect.objectContaining({
        connectionId: 'connection-a',
        method: 'PUT',
        path: '/api/config',
        profile: 'default'
      })
    )
    expect(puts[0][0].body).toEqual({
      config: record,
      write_mode: 'explicit_patch',
      unset_paths: []
    })
    expect(puts.filter(call => call[0].connectionId === 'connection-b')).toHaveLength(0)
  })
})
