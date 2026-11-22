export interface ConfigFieldSchema {
  category?: string
  description?: string
  options?: unknown[]
  /** When true, renders a SearchableSelect (Popover + cmdk) instead of the
   *  closed `<Select>` dropdown. For large option lists like IANA timezones. */
  searchable?: boolean
  /** A list field has a plugin-provided effective value; its editor shows the raw user contribution. */
  plugin_default?: boolean
  /** Read-time plugin merge mode; lists replace by default. */
  merge_mode?: 'append_unique' | 'replace'
  merge_description?: string
  /** When true, a searchable select prepends a "clear" item that resets the
   *  value to ''. Matches the existing <Select> EMPTY_SELECT_VALUE pattern. */
  clearable?: boolean
  type?: 'boolean' | 'list' | 'number' | 'select' | 'string' | 'text'
}
