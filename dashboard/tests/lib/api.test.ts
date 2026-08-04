import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  listNodes,
  listModels,
  submitModel,
  runInference,
  loadModel,
  redistributeModel,
  getModelStatus,
  getTopology,
} from '@/lib/api'

function mockFetchOnce(status: number, body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    })
  )
}

beforeEach(() => {
  vi.unstubAllGlobals()
})

describe('listNodes', () => {
  it('calls GET /nodes and returns the parsed body', async () => {
    mockFetchOnce(200, [{ node_id: 'node-1' }])

    const result = await listNodes()

    expect(fetch).toHaveBeenCalledWith(
      '/nodes',
      expect.objectContaining({ headers: expect.objectContaining({ 'Content-Type': 'application/json' }) })
    )
    expect(result).toEqual([{ node_id: 'node-1' }])
  })
})

describe('listModels', () => {
  it('calls GET /models and returns the parsed body', async () => {
    mockFetchOnce(200, [{ name: 'mamba-130m' }])

    const result = await listModels()

    expect(fetch).toHaveBeenCalledWith('/models', expect.anything())
    expect(result).toEqual([{ name: 'mamba-130m' }])
  })
})

describe('submitModel', () => {
  it('POSTs the manifest YAML and returns the registered model', async () => {
    mockFetchOnce(201, { name: 'mamba-130m' })

    const result = await submitModel('name: mamba-130m\n')

    expect(fetch).toHaveBeenCalledWith(
      '/models',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ manifest_yaml: 'name: mamba-130m\n' }),
      })
    )
    expect(result).toEqual({ name: 'mamba-130m' })
  })

  it('throws the backend detail message on a 400 response', async () => {
    mockFetchOnce(400, { detail: "Missing required field: checkpoint" })

    await expect(submitModel('name: broken\n')).rejects.toThrow(
      'Missing required field: checkpoint'
    )
  })

  it('throws the backend detail message on a 409 response', async () => {
    mockFetchOnce(409, { detail: "Model 'mamba-130m' is already registered" })

    await expect(submitModel('name: mamba-130m\n')).rejects.toThrow('already registered')
  })
})

describe('runInference', () => {
  it('POSTs model_name and input, returns the inference result', async () => {
    mockFetchOnce(200, {
      output: 'hello world',
      latency_ms: 123,
      node_latencies_ms: [50],
      peak_memory_mb: [260],
      num_nodes: 1,
      num_tokens: 20,
    })

    const result = await runInference('mamba-130m', 'hi')

    expect(fetch).toHaveBeenCalledWith(
      '/infer',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ model_name: 'mamba-130m', input: 'hi' }),
      })
    )
    expect(result.output).toBe('hello world')
    expect(result.num_nodes).toBe(1)
  })

  it('includes max_new_tokens in the body when provided', async () => {
    mockFetchOnce(200, {
      output: 'x',
      latency_ms: 1,
      node_latencies_ms: [],
      peak_memory_mb: [],
      num_nodes: 1,
      num_tokens: 5,
    })

    await runInference('mamba-130m', 'hi', 5)

    expect(fetch).toHaveBeenCalledWith(
      '/infer',
      expect.objectContaining({
        body: JSON.stringify({ model_name: 'mamba-130m', input: 'hi', max_new_tokens: 5 }),
      })
    )
  })

  it('throws the backend detail message on a 404 response', async () => {
    mockFetchOnce(404, { detail: "Model 'nonexistent' not found" })

    await expect(runInference('nonexistent', 'hi')).rejects.toThrow('not found')
  })
})

describe('loadModel', () => {
  it('POSTs to /models/{name}/load and returns the status', async () => {
    mockFetchOnce(200, { status: 'loading', error: null, num_nodes: 1 })

    const result = await loadModel('mamba-130m')

    expect(fetch).toHaveBeenCalledWith(
      '/models/mamba-130m/load',
      expect.objectContaining({ method: 'POST' })
    )
    expect(result.status).toBe('loading')
  })

  it('URL-encodes the model name', async () => {
    mockFetchOnce(200, { status: 'loading', error: null, num_nodes: 1 })

    await loadModel('weird name/x')

    expect(fetch).toHaveBeenCalledWith(
      '/models/weird%20name%2Fx/load',
      expect.anything()
    )
  })
})

describe('redistributeModel', () => {
  it('POSTs to /models/{name}/redistribute and returns the status', async () => {
    mockFetchOnce(200, { status: 'loading', error: null, num_nodes: 2 })

    const result = await redistributeModel('mamba-130m')

    expect(fetch).toHaveBeenCalledWith(
      '/models/mamba-130m/redistribute',
      expect.objectContaining({ method: 'POST' })
    )
    expect(result.status).toBe('loading')
  })
})

describe('getModelStatus', () => {
  it('calls GET /models/{name}/status and returns the parsed body', async () => {
    mockFetchOnce(200, { status: 'ready', error: null, num_nodes: 2 })

    const result = await getModelStatus('mamba-130m')

    expect(fetch).toHaveBeenCalledWith('/models/mamba-130m/status', expect.anything())
    expect(result).toEqual({ status: 'ready', error: null, num_nodes: 2 })
  })

  it('throws the backend detail message on a 404 response', async () => {
    mockFetchOnce(404, { detail: "Model 'nonexistent' not found" })

    await expect(getModelStatus('nonexistent')).rejects.toThrow('not found')
  })
})

describe('getTopology', () => {
  it('calls GET /topology and returns the parsed body', async () => {
    mockFetchOnce(200, {
      model_name: 'mamba-130m',
      assignments: [
        {
          node_id: 'node-0',
          ip_address: '192.168.1.10',
          layer_start: 0,
          layer_end: 24,
          is_first: true,
          is_last: true,
        },
      ],
    })

    const result = await getTopology()

    expect(fetch).toHaveBeenCalledWith('/topology', expect.anything())
    expect(result.model_name).toBe('mamba-130m')
    expect(result.assignments).toHaveLength(1)
  })
})
