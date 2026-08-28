import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, afterEach } from 'vitest'
import Home from '@/app/page'
import * as api from '@/lib/api'

afterEach(() => {
  vi.restoreAllMocks()
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

describe('Home (tab switching)', () => {
  it('keeps pages mounted across tab switches, preserving in-progress input', async () => {
    vi.spyOn(api, 'listNodes').mockResolvedValue([])
    vi.spyOn(api, 'listModels').mockResolvedValue([SAMPLE_MODEL])
    vi.spyOn(api, 'getTopology').mockResolvedValue({ model_name: null, assignments: [] })

    const user = userEvent.setup()
    render(<Home />)

    await user.click(screen.getByRole('button', { name: 'Inference' }))
    await waitFor(() =>
      expect(screen.getByRole('option', { name: 'dummy-mamba-tiny' })).toBeInTheDocument()
    )
    await user.selectOptions(screen.getByLabelText('Model'), 'dummy-mamba-tiny')

    // Both Models and Inference call listModels() once each on initial
    // mount (everything mounts eagerly, just hidden), so 2 calls total
    // is the correct baseline before any tab switching happens.
    const callsAfterFirstVisit = vi.mocked(api.listModels).mock.calls.length

    // listModels/loadModel aren't mocked to resolve to "ready", so the
    // prompt input stays disabled - type isn't possible there, but we can
    // still prove state survives by checking the model selection itself
    // persists across a tab switch instead of resetting.
    await user.click(screen.getByRole('button', { name: 'Stats' }))
    expect(screen.getByText('Total Nodes')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Inference' }))
    expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue('dummy-mamba-tiny')

    // Switching away and back must not re-trigger Inference's mount
    // effect - no new listModels() call, and the selection persisted.
    expect(api.listModels).toHaveBeenCalledTimes(callsAfterFirstVisit)
  })
})
