import { encodeConfigPath, getNestedValue, setNestedValue } from "./nested";

/** Build the raw patch from paths the user actually edited, including values equal to inherited defaults. */
export function buildExplicitConfigPatch(
  effectiveConfig: Record<string, unknown>,
  editedPaths: Iterable<string>,
): Record<string, unknown> {
  let patch: Record<string, unknown> = {};
  for (const path of editedPaths) {
    patch = setNestedValue(patch, path, getNestedValue(effectiveConfig, path));
  }
  return patch;
}

/** Overlay imported raw values onto the current raw/effective draft without expanding defaults. */
export function mergeConfigPatchPreview(
  base: Record<string, unknown>,
  patch: Record<string, unknown>,
): Record<string, unknown> {
  const result = structuredClone(base);
  for (const [key, value] of Object.entries(patch)) {
    encodeConfigPath([key]);
    const oldValue = Object.prototype.hasOwnProperty.call(result, key) ? result[key] : undefined;
    const merged = isRecord(value) && isRecord(oldValue)
      ? mergeConfigPatchPreview(oldValue, value)
      : structuredClone(value);
    Object.defineProperty(result, key, {
      value: merged,
      writable: true,
      enumerable: true,
      configurable: true,
    });
  }
  return result;
}

/** Paths intentionally supplied by an imported raw document. */
export function explicitConfigPaths(value: Record<string, unknown>): string[] {
  const paths: string[] = [];
  const visit = (node: Record<string, unknown>, prefix: string[] = []) => {
    for (const [key, child] of Object.entries(node)) {
      const path = [...prefix, key];
      encodeConfigPath([String(key)]);
      if (isRecord(child) && Object.keys(child).length > 0) visit(child, path);
      else paths.push(encodeConfigPath(path));
    }
  };
  visit(value);
  return paths;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
