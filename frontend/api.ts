export type ModelStatus = {
  ready: boolean
  status: string
  message: string
  model: { name?: string; family?: string; n_features?: number; n_train?: number; trained_at?: string; accuracy?: number } | null
  training: { running: boolean; phase: string; logs: string[]; error?: string | null }
  min_words: number
  max_characters: number
}

export type Rule = [string, number, number, number]
export type Analysis = {
  summary: {
    n_words: number; n_tokens: number; n_sentences: number; n_unique_words: number
    abstain: boolean; min_words: number; model_score: number | null; rule_score: number
    model_verdict: string | null; rule_verdict: string; elapsed_ms: number; model_note: string
  }
  lexer: { counts: Record<string, number>; tokens: { type: string; value: string; idx: number }[] }
  symbols: { unique_count: number; top_words: [string, number][] }
  parser: {
    tagger: string; n_sentences: number; parse_success_rate: number; mean_coverage: number
    mean_depth: number; mean_branching: number; mean_nodes: number
    structure_counts: Record<string, number>; template_counts: Record<string, number>
    first_tree: string; sentences: { tags: string[] }[]
  }
  features: { all: Record<string, number>; selected: Record<string, number> }
  rules: { score: number; total: number; max_total: number; fired: Rule[]; missed: Rule[] }
  stages: { id: string; name: string; output: string }[]
  console_output: string
}

export async function request<T>(path: string, options: RequestInit = {}, timeout = 30000): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeout)
  try {
    const response = await fetch(path, { ...options, signal: controller.signal, headers: { 'Content-Type': 'application/json', ...options.headers } })
    if (!response.headers.get('content-type')?.includes('application/json')) {
      throw new Error('The analysis server is unavailable. Check the terminal running npm run dev, then retry.')
    }
    const data = await response.json()
    if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : data.message || 'The request failed. Please retry.')
    return data as T
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw new Error('The request timed out. Your text is still here; please retry.')
    if (error instanceof TypeError) throw new Error('Cannot reach the local server. Check npm run dev is running, then retry.')
    throw error
  } finally { window.clearTimeout(timer) }
}
