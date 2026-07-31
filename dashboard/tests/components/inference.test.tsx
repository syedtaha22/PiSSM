import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, afterEach } from 'vitest'
import Inference from '@/components/pages/inference'
import * as api from '@/lib/api'
import { getInferenceLog } from '@/lib/history'

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
})

const SAMPLE_MODEL = {
  name: 'mamba-130m',
  arch: 'mamba',
  checkpoint: 'state-spaces/mamba-130m-hf',
  layers: 24,
  hidden_dim: 768,
  state_dim: 16,
  input_type: 'text',
  tokenizer: 'EleutherAI/gpt-neox-20b',
}

describe('Inference', () => {
  it('shows a loading message while the model preloads, disabling the input', async () => {
    vi.spyOn(api, 'listModels').mockResolvedValue([SAMPLE_MODEL])
    vi.spyOn(api, 'loadModel').mockResolvedValue({
      status: 'loading',
      error: null,
      num_nodes: 1,
    })
    vi.spyOn(api, 'getModelStatus').mockResolvedValue({
      status: 'loading',
      error: null,
      num_nodes: 1,
    })

    render(<Inference />)

    await waitFor(() =>
      expect(screen.getByRole('option', { name: 'mamba-130m' })).toBeInTheDocument()
    )
    await waitFor(() =>
      expect(screen.getByText(/Loading model onto 1 node/)).toBeInTheDocument()
    )
    expect(screen.getByPlaceholderText('Waiting for model to load...')).toBeDisabled()
  })

  it('enables the send button once the model reports ready, and a click sends the prompt', async () => {
    vi.spyOn(api, 'listModels').mockResolvedValue([SAMPLE_MODEL])
    vi.spyOn(api, 'loadModel').mockResolvedValue({
      status: 'ready',
      error: null,
      num_nodes: 1,
    })
    vi.spyOn(api, 'getModelStatus').mockResolvedValue({
      status: 'ready',
      error: null,
      num_nodes: 1,
    })
    vi.spyOn(api, 'runInference').mockResolvedValue({
      output: 'hello world',
      latency_ms: 120,
      node_latencies_ms: [50],
      peak_memory_mb: [260],
      num_nodes: 1,
    })

    const user = userEvent.setup()
    render(<Inference />)

    const input = await waitFor(() => {
      const el = screen.getByPlaceholderText('Enter prompt...')
      expect(el).not.toBeDisabled()
      return el
    })

    await user.type(input, 'hi there')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => expect(screen.getByText('hello world')).toBeInTheDocument())
    expect(api.runInference).toHaveBeenCalledWith('mamba-130m', 'hi there')

    // The input re-enables and regains focus once sending finishes,
    // instead of leaving the user having to re-click it each time.
    await waitFor(() => expect(input).not.toBeDisabled())
    await waitFor(() => expect(input).toHaveFocus())

    // The completed request's latency is logged for the Dashboard's chart.
    expect(getInferenceLog()).toEqual([
      { timestamp: expect.any(Number), modelName: 'mamba-130m', latencyMs: 120, numNodes: 1 },
    ])
  })

  it('clicking Redistribute re-dispatches across the current nodes', async () => {
    vi.spyOn(api, 'listModels').mockResolvedValue([SAMPLE_MODEL])
    vi.spyOn(api, 'loadModel').mockResolvedValue({
      status: 'ready',
      error: null,
      num_nodes: 1,
    })
    vi.spyOn(api, 'getModelStatus').mockResolvedValue({
      status: 'ready',
      error: null,
      num_nodes: 1,
    })
    const redistributeSpy = vi.spyOn(api, 'redistributeModel').mockResolvedValue({
      status: 'ready',
      error: null,
      num_nodes: 2,
    })

    const user = userEvent.setup()
    render(<Inference />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /redistribute/i })).not.toBeDisabled()
    )

    await user.click(screen.getByRole('button', { name: /redistribute/i }))

    await waitFor(() => expect(redistributeSpy).toHaveBeenCalledWith('mamba-130m'))
    expect(api.loadModel).toHaveBeenCalledTimes(1)
  })

  it('shows an error message when the model fails to load', async () => {
    vi.spyOn(api, 'listModels').mockResolvedValue([SAMPLE_MODEL])
    vi.spyOn(api, 'loadModel').mockResolvedValue({
      status: 'error',
      error: 'no available nodes in the registry',
      num_nodes: null,
    })
    vi.spyOn(api, 'getModelStatus').mockResolvedValue({
      status: 'error',
      error: 'no available nodes in the registry',
      num_nodes: null,
    })

    render(<Inference />)

    await waitFor(() =>
      expect(
        screen.getByText(/Failed to load model: no available nodes/)
      ).toBeInTheDocument()
    )
  })

  it('shows a message when no models are registered', async () => {
    vi.spyOn(api, 'listModels').mockResolvedValue([])

    render(<Inference />)

    await waitFor(() =>
      expect(
        screen.getByText('No models registered. Upload a manifest on the Models page.')
      ).toBeInTheDocument()
    )
  })
})
