const UNSAFE_CONFIG_PATH_PARTS = new Set(['__proto__', 'constructor', 'prototype'])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function assertSafePart(part: string): void {
  if (!part || UNSAFE_CONFIG_PATH_PARTS.has(part)) {
    throw new Error(`Unsafe config path part: ${part}`)
  }
}

/** Split config paths using the backend's backslash-escaped literal-dot convention. */
export function configPathParts(path: string): string[] {
  const parts: string[] = []
  let current = ''
  for (let index = 0; index < path.length; index += 1) {
    const char = path[index]
    if (char === '\\' && path[index + 1] === '.') {
      current += '.'
      index += 1
    } else if (char === '.') {
      assertSafePart(current)
      parts.push(current)
      current = ''
    } else {
      current += char
    }
  }
  assertSafePart(current)
  parts.push(current)
  return parts
}

/** Encode segments for the backend's dotted-path wire format. */
export function encodeConfigPath(parts: readonly string[]): string {
  parts.forEach(assertSafePart)
  return parts.map(part => part.replace(/\./g, '\\.')).join('.')
}

/** True when one explicit config path is the same as or an ancestor of the other. */
export function configPathsOverlap(left: string, right: string): boolean {
  const leftParts = configPathParts(left)
  const rightParts = configPathParts(right)
  const sharedLength = Math.min(leftParts.length, rightParts.length)
  return leftParts.slice(0, sharedLength).every((part, index) => part === rightParts[index])
}

function safeSet(target: Record<string, unknown>, key: string, value: unknown): void {
  assertSafePart(key)
  Object.defineProperty(target, key, {
    value,
    writable: true,
    enumerable: true,
    configurable: true,
  })
}

export function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
  const parts = configPathParts(path)
  let current: unknown = obj
  for (const part of parts) {
    if (!isRecord(current) || !Object.prototype.hasOwnProperty.call(current, part)) return undefined
    current = current[part]
  }
  return current
}

export function unsetNestedValue(obj: Record<string, unknown>, path: string): Record<string, unknown> {
  const clone = structuredClone(obj)
  const parts = configPathParts(path)
  let current: Record<string, unknown> = clone
  for (const part of parts.slice(0, -1)) {
    if (!Object.prototype.hasOwnProperty.call(current, part) || !isRecord(current[part])) return clone
    current = current[part] as Record<string, unknown>
  }
  if (Object.prototype.hasOwnProperty.call(current, parts[parts.length - 1])) {
    delete current[parts[parts.length - 1]]
  }
  return clone
}

export function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const clone = structuredClone(obj)
  const parts = configPathParts(path)
  let current: Record<string, unknown> = clone
  for (const part of parts.slice(0, -1)) {
    const existing = Object.prototype.hasOwnProperty.call(current, part) ? current[part] : undefined
    if (!isRecord(existing)) safeSet(current, part, {})
    current = current[part] as Record<string, unknown>
  }
  safeSet(current, parts[parts.length - 1], value)
  return clone
}
