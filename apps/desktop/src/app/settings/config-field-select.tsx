import type { ReactNode } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { prettyName } from '@/lib/text'
import type { ConfigFieldSchema } from '@/types/config-field-schema'

import { ComboboxInput } from './combobox-input'
import { CONTROL_TEXT, EMPTY_SELECT_VALUE, FREE_INPUT_KEYS } from './constants'
import { SearchableSelect } from './searchable-select'

type ConfigFieldRow = (action: ReactNode) => ReactNode

interface ConfigFieldSelectCopy {
  builtinOnly: string
  noResults: string
  none: string
  noneParen: string
  notSet: string
  searchPlaceholder: string
  systemDefault: string
}

interface ConfigFieldSelectProps {
  copy: ConfigFieldSelectCopy
  onChange: (value: unknown) => void
  optionLabels?: Record<string, string>
  options?: string[]
  renderRow: ConfigFieldRow
  schema: ConfigFieldSchema
  schemaKey: string
  value: unknown
}

export function renderConfigFieldSelect({
  copy,
  onChange,
  optionLabels,
  options,
  renderRow,
  schema,
  schemaKey,
  value
}: ConfigFieldSelectProps): ReactNode | undefined {
  if (!options) {
    return undefined
  }

  if (schema.searchable) {
    return renderRow(
      <SearchableSelect
        clearLabel={schema.clearable ? copy.systemDefault : undefined}
        emptyMessage={copy.noResults}
        onChange={onChange}
        options={options.filter(option => option !== '')}
        placeholder={copy.searchPlaceholder}
        value={String(value ?? '')}
      />
    )
  }

  if (FREE_INPUT_KEYS.has(schemaKey)) {
    return renderRow(
      <ComboboxInput
        className={CONTROL_TEXT}
        onChange={onChange}
        optionLabels={optionLabels}
        options={options.filter(option => option !== '')}
        placeholder={copy.notSet}
        value={String(value ?? '')}
      />
    )
  }

  const emptyLabel = schemaKey === 'display.personality'
    ? copy.none
    : schemaKey === 'memory.provider'
      ? copy.builtinOnly
      : copy.noneParen

  return renderRow(
    <Select
      onValueChange={next => onChange(next === EMPTY_SELECT_VALUE ? '' : next)}
      value={String(value ?? '') || EMPTY_SELECT_VALUE}
    >
      <SelectTrigger className={CONTROL_TEXT}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map(option => (
          <SelectItem key={option || EMPTY_SELECT_VALUE} value={option || EMPTY_SELECT_VALUE}>
            {option ? (optionLabels?.[option] ?? prettyName(option)) : emptyLabel}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
