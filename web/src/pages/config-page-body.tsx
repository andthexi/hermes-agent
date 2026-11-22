import type { ComponentType, ReactNode } from 'react'

import { FileText, Filter, Search } from 'lucide-react'

import { Button } from '@nous-research/ui/ui/components/button'
import { Badge } from '@nous-research/ui/ui/components/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@nous-research/ui/ui/components/card'
import { ListItem } from '@nous-research/ui/ui/components/list-item'
import { Spinner } from '@nous-research/ui/ui/components/spinner'
import { useI18n } from '@/i18n'
import type { ConfigYamlStatus } from '@/lib/use-config-yaml-source'

interface ConfigPageBodyProps {
  CategoryIcon: ComponentType<{ category: string; className?: string }>
  activeCategory: string
  activeFields: [string, Record<string, unknown>][]
  categories: string[]
  categoryCounts: Record<string, number>
  isSearching: boolean
  onRetryYamlSource: () => void
  onSelectCategory: (category: string) => void
  onYamlChange: (text: string) => void
  prettyCategoryName: (category: string) => string
  renderFields: (fields: [string, Record<string, unknown>][], showCategory?: boolean) => ReactNode
  searchMatchedFields: [string, Record<string, unknown>][]
  searchQuery: string
  yamlLoading: boolean
  yamlMode: boolean
  yamlStatus: ConfigYamlStatus
  yamlText: string
}

export function ConfigPageBody({
  CategoryIcon,
  activeCategory,
  activeFields,
  categories,
  categoryCounts,
  isSearching,
  onRetryYamlSource,
  onSelectCategory,
  onYamlChange,
  prettyCategoryName,
  renderFields,
  searchMatchedFields,
  searchQuery,
  yamlLoading,
  yamlMode,
  yamlStatus,
  yamlText
}: ConfigPageBodyProps) {
  const { t } = useI18n()

  if (yamlMode) {
    return (
      <Card>
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm flex items-center gap-2">
            <FileText className="h-4 w-4" />
            {t.config.rawYaml}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {yamlLoading ? (
            <div className="flex items-center justify-center py-12">
              <Spinner className="text-xl text-primary" />
            </div>
          ) : yamlStatus === 'error' ? (
            <div className="flex items-center justify-between gap-3 p-4" role="alert">
              <span>{t.config.failedToLoadRaw}</span>
              <Button ghost size="sm" onClick={onRetryYamlSource}>{t.common.refresh}</Button>
            </div>
          ) : (
            <textarea
              className="flex min-h-[600px] w-full bg-transparent px-4 py-3 text-sm font-mono leading-relaxed placeholder:text-muted-foreground focus-visible:outline-none border-t border-border"
              value={yamlText}
              onChange={event => onYamlChange(event.target.value)}
              spellCheck={false}
              disabled={yamlStatus !== 'ready'}
            />
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="flex flex-col sm:flex-row gap-4">
      <aside aria-label={t.config.filters} className="sm:w-56 sm:shrink-0">
        <div className="sm:sticky sm:top-4">
          <div className="flex flex-col border border-border bg-muted/20">
            <div className="hidden sm:flex items-center gap-2 px-3 py-2 border-b border-border">
              <Filter className="h-3 w-3 text-text-tertiary" />
              <span className="font-mondwest text-display text-xs tracking-[0.12em] text-text-secondary">
                {t.config.filters}
              </span>
            </div>

            <div className="hidden sm:block px-3 pt-2 pb-1 font-mondwest text-display text-xs tracking-[0.12em] text-text-tertiary">
              {t.config.sections}
            </div>

            <div className="flex sm:flex-col gap-1 sm:gap-px p-2 sm:pt-1 overflow-x-auto sm:overflow-x-visible scrollbar-none sm:max-h-[calc(100vh-260px)] sm:overflow-y-auto">
              {categories.map(category => {
                const isActive = !isSearching && activeCategory === category

                return (
                  <ListItem
                    key={category}
                    active={isActive}
                    onClick={() => onSelectCategory(category)}
                    className="rounded-none whitespace-nowrap px-2 py-1 text-xs"
                  >
                    <CategoryIcon category={category} className="h-3.5 w-3.5 shrink-0" />
                    <span className="flex-1 truncate">{prettyCategoryName(category)}</span>
                    <span className={`text-xs tabular-nums ${isActive ? 'text-text-secondary' : 'text-text-tertiary'}`}>
                      {categoryCounts[category] || 0}
                    </span>
                  </ListItem>
                )
              })}
            </div>
          </div>
        </div>
      </aside>

      <div className="flex-1 min-w-0">
        {isSearching ? (
          <Card>
            <CardHeader className="py-3 px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Search className="h-4 w-4" />
                  {t.config.searchResults}
                </CardTitle>
                <Badge tone="secondary" className="text-xs">
                  {searchMatchedFields.length} {t.config.fields.replace('{s}', searchMatchedFields.length !== 1 ? 's' : '')}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="grid gap-2 px-4 pb-4">
              {searchMatchedFields.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                  {t.config.noFieldsMatch.replace('{query}', searchQuery)}
                </p>
              ) : renderFields(searchMatchedFields, true)}
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader className="py-3 px-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm flex items-center gap-2">
                  <CategoryIcon category={activeCategory} className="h-4 w-4" />
                  {prettyCategoryName(activeCategory)}
                </CardTitle>
                <Badge tone="secondary" className="text-xs">
                  {activeFields.length} {t.config.fields.replace('{s}', activeFields.length !== 1 ? 's' : '')}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="grid gap-2 px-4 pb-4">{renderFields(activeFields)}</CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
