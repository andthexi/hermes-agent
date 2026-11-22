type ConfigFetch = <T>(url: string, init?: RequestInit) => Promise<T>
type ProfileUrl = (url: string, profile?: string) => string

export function createConfigApi(
  fetchJSON: ConfigFetch,
  appendProfileParam: ProfileUrl,
  getManagementProfile: () => string,
) {
  return {
    getConfig: (profile = getManagementProfile()) =>
      fetchJSON<Record<string, unknown>>(appendProfileParam('/api/config', profile)),
    getConfigSource: (profile = getManagementProfile()) =>
      fetchJSON<Record<string, unknown>>(appendProfileParam('/api/config?raw_source=true', profile)),
    getDefaults: () => fetchJSON<Record<string, unknown>>('/api/config/defaults'),
    getSchema: (profile = getManagementProfile()) =>
      fetchJSON<{ fields: Record<string, unknown>; category_order: string[] }>(appendProfileParam('/api/config/schema', profile)),
    getConfigRaw: (profile = getManagementProfile()) =>
      fetchJSON<{ yaml: string; path?: string }>(appendProfileParam('/api/config/raw', profile)),
    saveConfig: (
      config: Record<string, unknown>,
      profile = getManagementProfile(),
      unsetPaths: string[] = [],
    ) =>
      fetchJSON<{ ok: boolean }>(appendProfileParam('/api/config', profile), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config, write_mode: 'explicit_patch', unset_paths: unsetPaths }),
      }),
    saveConfigRaw: (yaml_text: string, profile = getManagementProfile()) =>
      fetchJSON<{ ok: boolean }>(appendProfileParam('/api/config/raw', profile), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml_text }),
      }),
  }
}
