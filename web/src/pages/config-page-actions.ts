import type { ChangeEvent, Dispatch, SetStateAction } from 'react'

import { configPathsOverlap, getNestedValue, setNestedValue, unsetNestedValue } from '@/lib/nested'
import { explicitConfigPaths, mergeConfigPatchPreview } from '@/lib/config-write'

interface ConfigPageActionsInput {
  activeCategory: string
  activeFields: [string, Record<string, unknown>][]
  config: Record<string, unknown> | null
  defaults: Record<string, unknown> | null
  isSearching: boolean
  rawConfig: Record<string, unknown> | null
  rawSourceReady: boolean
  configImported: string
  invalidJson: string
  resetScopeToast: string
  searchMatchedFields: [string, Record<string, unknown>][]
  schema: Record<string, Record<string, unknown>> | null
  searchResults: string
  setConfig: Dispatch<SetStateAction<Record<string, unknown> | null>>
  setConfirmReset: Dispatch<SetStateAction<boolean>>
  setEditedPaths: Dispatch<SetStateAction<string[]>>
  setRawConfig: Dispatch<SetStateAction<Record<string, unknown> | null>>
  setUnsetPaths: Dispatch<SetStateAction<string[]>>
  showToast: (message: string, kind: 'error' | 'success') => void
  prettyCategoryName: (category: string) => string
}

export function createConfigPageActions({
  activeCategory,
  activeFields,
  config,
  defaults,
  isSearching,
  rawConfig,
  rawSourceReady,
  configImported,
  invalidJson,
  resetScopeToast,
  searchMatchedFields,
  schema,
  searchResults,
  setConfig,
  setConfirmReset,
  setEditedPaths,
  setRawConfig,
  setUnsetPaths,
  showToast,
  prettyCategoryName
}: ConfigPageActionsInput) {
  const scopedFields = () => isSearching ? searchMatchedFields : activeFields

  const handleReset = () => {
    if (!defaults || !config || !rawSourceReady) return
    if (scopedFields().length === 0) return
    setConfirmReset(true)
  }

  const executeReset = () => {
    if (!defaults || !config || !rawSourceReady) return
    setConfirmReset(false)
    const fields = scopedFields()
    if (fields.length === 0) return

    const scopeLabel = isSearching ? searchResults : prettyCategoryName(activeCategory)
    const resetPaths = fields.map(([key]) => key)
    setUnsetPaths(previous => [...new Set([...previous, ...resetPaths])])
    setEditedPaths(previous => previous.filter(path => !resetPaths.includes(path)))
    setRawConfig(previous => {
      const source = previous ?? rawConfig
      if (!source) return previous
      let next = source
      for (const path of resetPaths) next = unsetNestedValue(next, path)
      return next
    })

    let nextConfig = config
    for (const path of resetPaths) {
      if (!schema?.[path]?.plugin_default) {
        nextConfig = setNestedValue(nextConfig, path, getNestedValue(defaults, path))
      }
    }
    setConfig(nextConfig)
    showToast(resetScopeToast.replace('{scope}', scopeLabel), 'success')
  }

  const handleExport = () => {
    if (!rawSourceReady || !rawConfig) return
    const blob = new Blob([JSON.stringify(rawConfig, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'hermes-config.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  const handleImport = (event: ChangeEvent<HTMLInputElement>) => {
    if (!rawSourceReady || !rawConfig) {
      event.target.value = ''
      return
    }
    const file = event.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const imported: unknown = JSON.parse(String(reader.result))
        if (typeof imported !== 'object' || imported === null || Array.isArray(imported)) {
          throw new Error('Config import must be an object')
        }
        const importedConfig = imported as Record<string, unknown>
        const importedPaths = explicitConfigPaths(importedConfig)
        setConfig(previous => mergeConfigPatchPreview(previous ?? {}, importedConfig))
        setRawConfig(previous => {
          const source = previous ?? rawConfig
          return source ? mergeConfigPatchPreview(source, importedConfig) : previous
        })
        setEditedPaths(previous => [...new Set([...previous, ...importedPaths])])
        setUnsetPaths(previous => previous.filter(
          path => !importedPaths.some(importedPath => configPathsOverlap(path, importedPath)),
        ))
        showToast(configImported, 'success')
      } catch {
        showToast(invalidJson, 'error')
      }
    }
    reader.readAsText(file)
    event.target.value = ''
  }

  return { executeReset, handleExport, handleImport, handleReset }
}
