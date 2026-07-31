const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? ''

export interface NodeSummary {
  node_id: string
  ip_address: string
  available_ram_mb: number
  total_ram_mb: number
  cpu_count: number
  arch: string
  os_name: string
  os_version: string
  status: string
  last_heartbeat: number
  first_seen: number
  inference_port: number
}

export interface ModelSummary {
  name: string
  arch: string
  checkpoint: string
  layers: number
  hidden_dim: number
  state_dim: number
  input_type: string
  tokenizer: string
}

export interface InferResult {
  output: string
  latency_ms: number
  node_latencies_ms: number[]
  peak_memory_mb: number[]
  num_nodes: number
}

export type ModelLoadStatus = 'not_loaded' | 'loading' | 'ready' | 'error'

export interface ModelStatus {
  status: ModelLoadStatus
  error: string | null
  num_nodes: number | null
}

export interface TopologyAssignment {
  node_id: string
  ip_address: string
  layer_start: number
  layer_end: number
  is_first: boolean
  is_last: boolean
}

export interface Topology {
  model_name: string | null
  assignments: TopologyAssignment[]
}

class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const message = body?.detail ?? `Request to ${path} failed with ${response.status}`
    throw new ApiError(response.status, message)
  }

  return response.json() as Promise<T>
}

export async function listNodes(): Promise<NodeSummary[]> {
  return request<NodeSummary[]>('/nodes')
}

export async function listModels(): Promise<ModelSummary[]> {
  return request<ModelSummary[]>('/models')
}

export async function submitModel(manifestYaml: string): Promise<ModelSummary> {
  return request<ModelSummary>('/models', {
    method: 'POST',
    body: JSON.stringify({ manifest_yaml: manifestYaml }),
  })
}

export async function runInference(
  modelName: string,
  input: string,
  maxNewTokens?: number
): Promise<InferResult> {
  return request<InferResult>('/infer', {
    method: 'POST',
    body: JSON.stringify({
      model_name: modelName,
      input,
      ...(maxNewTokens !== undefined && { max_new_tokens: maxNewTokens }),
    }),
  })
}

export async function loadModel(modelName: string): Promise<ModelStatus> {
  return request<ModelStatus>(`/models/${encodeURIComponent(modelName)}/load`, {
    method: 'POST',
  })
}

export async function getModelStatus(modelName: string): Promise<ModelStatus> {
  return request<ModelStatus>(`/models/${encodeURIComponent(modelName)}/status`)
}

export async function redistributeModel(modelName: string): Promise<ModelStatus> {
  return request<ModelStatus>(`/models/${encodeURIComponent(modelName)}/redistribute`, {
    method: 'POST',
  })
}

export async function getTopology(): Promise<Topology> {
  return request<Topology>('/topology')
}
