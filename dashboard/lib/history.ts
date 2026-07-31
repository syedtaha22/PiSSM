const STORAGE_KEY = 'pissm-inference-history'
const MAX_ENTRIES = 50

export interface InferenceLogEntry {
  timestamp: number
  modelName: string
  latencyMs: number
  numNodes: number
}

export function appendInferenceLog(entry: InferenceLogEntry): void {
  if (typeof window === 'undefined') return
  try {
    const next = [...getInferenceLog(), entry].slice(-MAX_ENTRIES)
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  } catch {
    /* localStorage unavailable (private browsing, quota) - history just won't persist */
  }
}

export function getInferenceLog(): InferenceLogEntry[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}
