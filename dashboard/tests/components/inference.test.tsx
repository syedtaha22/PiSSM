import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, afterEach } from 'vitest'
import Inference from '@/components/pages/inference'
import * as api from '@/lib/api'
import { getInferenceLog } from '@/lib/history'

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
  window.localStorage.clear()
})

const SAMPLE_MODEL = {
  name: 'dummy-mamba-tiny',
  arch: 'mamba',
  checkpoint: 'checkpoints/dummy-mamba-tiny',
  layers: 4,
  hidden_dim: 64,
  state_dim: 8,
  input_type: 'text',
  tokenizer: 'EleutherAI/gpt-neox-20b',
}

async function selectModel(user: ReturnType<typeof userEvent.setup>) {
  await waitFor(() =>
    expect(screen.getByRole('option', { name: 'dummy-mamba-tiny' })).toBeInTheDocument()
  )
  await user.selectOptions(screen.getByLabelText('Model'), 'dummy-mamba-tiny')
}

describe('Inference', () => {
  it('does not load any model just from listing them - loading is explicit', async () => {
    vi.spyOn(api, 'listModels').mockResolvedValue([SAMPLE_MODEL])
    const loadSpy = vi.spyOn(api, 'loadModel').mockResolvedValue({
      status: 'ready',
      error: null,
      num_nodes: 1,
    })

    render(<Inference />)

    await waitFor(() =>
      expect(screen.getByRole('option', { name: 'dummy-mamba-tiny' })).toBeInTheDocument()
    )
    // Give any errant auto-load effect a chance to fire before asserting
    // it didn't.
    await new Promise((r) => setTimeout(r, 50))

    expect(loadSpy).not.toHaveBeenCalled()
    expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('')
  })

  it('polls listModels so a model registered elsewhere eventually appears without a remount', async () => {
    // Pages stay mounted for the whole session now (app/page.tsx keeps
    // every tab alive), so Inference can no longer rely on a tab-switch
    // remount to pick up a model someone just uploaded on the Models
    // page - it has to notice on its own.
    vi.useFakeTimers()
    const listModelsSpy = vi
      .spyOn(api, 'listModels')
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([SAMPLE_MODEL])

    render(<Inference />)
    await vi.advanceTimersByTimeAsync(0)

    expect(listModelsSpy).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('option', { name: 'dummy-mamba-tiny' })).not.toBeInTheDocument()

    // Matches MODELS_POLL_INTERVAL_MS in inference.tsx.
    await vi.advanceTimersByTimeAsync(3000)

    expect(listModelsSpy).toHaveBeenCalledTimes(2)
    expect(screen.getByRole('option', { name: 'dummy-mamba-tiny' })).toBeInTheDocument()
  })

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

    const user = userEvent.setup()
    render(<Inference />)
    await selectModel(user)

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
    vi.spyOn(api, 'runInferenceStream').mockImplementation(
      async (_modelName, _input, onToken) => {
        onToken('hello ')
        onToken('world')
        return {
          output: 'hello world',
          latency_ms: 120,
          node_latencies_ms: [50],
          peak_memory_mb: [260],
          num_nodes: 1,
          num_tokens: 20,
        }
      }
    )

    const user = userEvent.setup()
    render(<Inference />)
    await selectModel(user)

    const input = await waitFor(() => {
      const el = screen.getByPlaceholderText('Enter prompt...')
      expect(el).not.toBeDisabled()
      return el
    })

    await user.type(input, 'hi there')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => expect(screen.getByText('hello world')).toBeInTheDocument())
    expect(api.runInferenceStream).toHaveBeenCalledWith(
      'dummy-mamba-tiny',
      'hi there',
      expect.any(Function)
    )

    // The input re-enables and regains focus once sending finishes,
    // instead of leaving the user having to re-click it each time.
    await waitFor(() => expect(input).not.toBeDisabled())
    await waitFor(() => expect(input).toHaveFocus())

    // The completed request's latency is logged for the Dashboard's chart.
    expect(getInferenceLog()).toEqual([
      {
        timestamp: expect.any(Number),
        modelName: 'dummy-mamba-tiny',
        latencyMs: 120,
        numNodes: 1,
        numTokens: 20,
      },
    ])

    // 20 tokens in 120ms = 166.7 tok/s.
    expect(screen.getByText('166.7 tok/s')).toBeInTheDocument()
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
    await selectModel(user)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /redistribute/i })).not.toBeDisabled()
    )

    await user.click(screen.getByRole('button', { name: /redistribute/i }))

    await waitFor(() => expect(redistributeSpy).toHaveBeenCalledWith('dummy-mamba-tiny'))
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

    const user = userEvent.setup()
    render(<Inference />)
    await selectModel(user)

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
