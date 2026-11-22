import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ConfigApi from '@/api/config'
import { $settingsRequestProfile } from '@/store/settings-scope'

import type { ConfigSettings as ConfigSettingsType } from './config-settings'

// The vi.mock factory below replaces the computed (read-only) atom with a
// writable one; narrow the import back so tests can drive it.
const scopeProfileMock = $settingsRequestProfile as unknown as { set: (value: string) => void }

const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const saveHermesConfig = vi.fn()
const getElevenLabsVoices = vi.fn()

// Keep the real read-origin helpers (WeakMap peek/bind) live: the shared
// config hook reaches them through the barrel, and a bare mock would throw.
vi.mock('@/hermes', async () => ({
  ...(await vi.importActual<typeof ConfigApi>('@/api/config')),
  // use-config-record folds the scope into its cache key via the barrel; the
  // real one is a pure string fold, mirrored here for the string scopes this
  // suite passes.
  profileScopeKey: (scope?: unknown) => (typeof scope === 'string' && scope.trim()) || 'default',
  getHermesConfigRecord: (profile?: string, options?: { rawSource?: boolean }) => getHermesConfigRecord(profile, options),
  getHermesConfigSchema: () => getHermesConfigSchema(),
  saveHermesConfig: (config: unknown, profile?: string) => saveHermesConfig(config, profile),
  getElevenLabsVoices: () => getElevenLabsVoices(),
  setApiRequestProfile: () => {}
}))

vi.mock('../hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: () => {}
}))

// The real stores pull in the gateway/profile stack, which needs a live
// backend connection. This page only reads the "applies to" scope override
// and the repo-discovery signature, neither of which this test touches. The
// scope chip it renders also reads the selected profile and the loud-note
// selector, so those are stubbed to the single-profile default shape.
vi.mock('@/store/settings-scope', () => ({
  // The real store derives this from the displayed $settingsScopeProfile
  // (never undefined for a real profile — settings-scope.test.ts pins that);
  // here it is a plain atom so the page's threading of it can be driven.
  $settingsRequestProfile: atom<string | undefined>('default'),
  $settingsScopeEditsNonDefault: atom(false),
  $settingsScopeOverride: atom<null | string>(null),
  $settingsScopeProfile: atom<string>('default')
}))

vi.mock('@/store/projects', () => ({
  $worktreeRefreshToken: atom(0),
  repoDiscoveryPolicyFromConfig: () => ({ enabled: true, roots: [], exclude_paths: [] }),
  repoDiscoveryPolicySignature: (policy: unknown) => JSON.stringify(policy),
  scanAndRecordRepos: vi.fn().mockResolvedValue(undefined)
}))

// The module graph behind ConfigSettings is large (1.5s cold here, >10s on a
// saturated CI runner); load it once under the hook timeout so the 15s test
// budget is spent on the autosave behaviour, not on transform + import.
let ConfigSettings: typeof ConfigSettingsType

beforeAll(async () => {
  ;({ ConfigSettings } = await import('./config-settings'))
}, 60_000)

beforeEach(() => {
  scopeProfileMock.set('default')
  getElevenLabsVoices.mockResolvedValue({ available: false })
  getHermesConfigSchema.mockResolvedValue({ fields: {} })
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, reject, resolve }
}

function renderConfigSettings(activeSectionId = 'safety') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const importInputRef = createRef<HTMLInputElement>()

  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ConfigSettings activeSectionId={activeSectionId} importInputRef={importInputRef} />
      </QueryClientProvider>
    </MemoryRouter>
  )

  return { importInputRef }
}

describe('ConfigSettings autosave', () => {
  it('sends a later revert instead of diffing it away against the stale page-load baseline', async () => {
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false }, other: 'untouched' })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings()

      const toggle = await screen.findByRole('switch')

      // Edit: flip checkpoints.enabled on, let the debounced autosave fire.
      toggle.click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(1))
      expect(saveHermesConfig.mock.calls[0][0]).toEqual({ checkpoints: { enabled: true } })

      // Revert: flip it back to its original value and let autosave fire again.
      toggle.click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(2))
      // Must still explicitly send the reverted value — diffing against the
      // never-advanced page-load baseline would produce an empty patch here
      // (the field is back to its original value) and leave disk stuck at
      // `enabled: true` from the first save.
      expect(saveHermesConfig.mock.calls[1][0]).toEqual({ checkpoints: { enabled: false } })
    } finally {
      vi.useRealTimers()
    }
  })

  it('edits an append_unique list from raw profile entries only', async () => {
    const effective = { desktop: { repo_scan_roots: ['user-root', 'shared-root'] } }
    const raw = { desktop: { repo_scan_roots: ['user-root'] } }
    getHermesConfigRecord.mockImplementation((_profile, options) =>
      options?.rawSource ? Promise.resolve(raw) : Promise.resolve(effective)
    )
    getHermesConfigSchema.mockResolvedValue({
      fields: { 'desktop.repo_scan_roots': { type: 'list', category: 'workspace', plugin_default: true, merge_mode: 'append_unique' } }
    })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings('workspace')

      const field = await screen.findByRole('textbox')
      expect((field as HTMLInputElement).value).toBe('user-root')
      fireEvent.focus(field)
      fireEvent.change(field, { target: { value: 'user-root, another-root' } })
      fireEvent.blur(field)
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(1))
      expect(saveHermesConfig.mock.calls[0][0]).toEqual({ desktop: { repo_scan_roots: ['user-root', 'another-root'] } })
    } finally {
      vi.useRealTimers()
    }
  })

  it('keeps append_unique editing unavailable until a failed raw read is retried', async () => {
    const effective = { desktop: { repo_scan_roots: ['user-root', 'shared-root'] } }
    const firstRaw = deferred<{ desktop: { repo_scan_roots: string[] } }>()
    const retriedRaw = deferred<{ desktop: { repo_scan_roots: string[] } }>()
    let rawAttempts = 0
    getHermesConfigRecord.mockImplementation((_profile, options) => {
      if (!options?.rawSource) {
        return Promise.resolve(effective)
      }
      rawAttempts += 1
      return rawAttempts === 1 ? firstRaw.promise : retriedRaw.promise
    })
    getHermesConfigSchema.mockResolvedValue({
      fields: { 'desktop.repo_scan_roots': { type: 'list', category: 'workspace', plugin_default: true, merge_mode: 'append_unique' } }
    })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      const { importInputRef } = renderConfigSettings('workspace')

      const field = await screen.findByRole('textbox') as HTMLInputElement
      expect(field.disabled).toBe(true)
      fireEvent.change(field, { target: { value: 'invented-root' } })
      fireEvent.blur(field)
      await vi.advanceTimersByTimeAsync(700)
      expect(saveHermesConfig).not.toHaveBeenCalled()

      await act(async () => firstRaw.reject(new Error('raw source unavailable')))
      expect(await screen.findByRole('alert')).toBeTruthy()
      expect(field.disabled).toBe(true)
      expect(importInputRef.current?.disabled).toBe(true)

      fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
      await vi.waitFor(() => expect(rawAttempts).toBe(2))
      await act(async () => retriedRaw.resolve({ desktop: { repo_scan_roots: ['user-root'] } }))
      await vi.waitFor(() => expect(field.value).toBe('user-root'))
      expect(field.disabled).toBe(false)
      expect(importInputRef.current?.disabled).toBe(false)

      fireEvent.focus(field)
      fireEvent.change(field, { target: { value: 'user-root, another-root' } })
      fireEvent.blur(field)
      await vi.advanceTimersByTimeAsync(700)
      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(1))
      expect(saveHermesConfig.mock.calls[0][0]).toEqual({ desktop: { repo_scan_roots: ['user-root', 'another-root'] } })
    } finally {
      vi.useRealTimers()
    }
  })

  it('ignores a late raw response from the previous profile', async () => {
    const alphaRaw = deferred<{ desktop: { repo_scan_roots: string[] } }>()
    const betaRaw = deferred<{ desktop: { repo_scan_roots: string[] } }>()
    getHermesConfigRecord.mockImplementation((profile, options) => {
      if (!options?.rawSource) {
        return Promise.resolve({ desktop: { repo_scan_roots: [`${profile}-user`, `${profile}-shared`] } })
      }
      return profile === 'alpha' ? alphaRaw.promise : betaRaw.promise
    })
    getHermesConfigSchema.mockResolvedValue({
      fields: { 'desktop.repo_scan_roots': { type: 'list', category: 'workspace', plugin_default: true, merge_mode: 'append_unique' } }
    })

    renderConfigSettings('workspace')
    const alphaField = await screen.findByRole('textbox') as HTMLInputElement
    expect(alphaField.disabled).toBe(true)

    await act(async () => scopeProfileMock.set('beta'))
    await vi.waitFor(() => expect(getHermesConfigRecord).toHaveBeenCalledWith('beta', { rawSource: true }))
    const betaField = await screen.findByRole('textbox') as HTMLInputElement
    expect(betaField.disabled).toBe(true)

    await act(async () => alphaRaw.resolve({ desktop: { repo_scan_roots: ['alpha-user'] } }))
    expect(betaField.disabled).toBe(true)
    await act(async () => betaRaw.resolve({ desktop: { repo_scan_roots: ['beta-user'] } }))
    await vi.waitFor(() => expect(betaField.value).toBe('beta-user'))
    expect(betaField.disabled).toBe(false)
  })

  it('threads the "Applies to" request scope into both the config read and the autosave write', async () => {
    // #118432: the request scope is the concrete profile the page displays
    // (see settings-scope.test.ts). The page must carry it into the read AND
    // the write — a read scoped to B with a write that falls back to the
    // ambient (launch) profile is exactly the silent cross-profile write.
    scopeProfileMock.set('nash')
    getHermesConfigRecord.mockResolvedValue({ checkpoints: { enabled: false } })

    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      renderConfigSettings()

      await vi.waitFor(() => expect(getHermesConfigRecord).toHaveBeenCalledWith('nash', undefined))
      expect(getHermesConfigRecord).toHaveBeenCalledWith('nash', { rawSource: true })

      ;(await screen.findByRole('switch')).click()
      await vi.advanceTimersByTimeAsync(700)

      await vi.waitFor(() => expect(saveHermesConfig).toHaveBeenCalledWith({ checkpoints: { enabled: true } }, 'nash'))
    } finally {
      vi.useRealTimers()
    }
  })
})
