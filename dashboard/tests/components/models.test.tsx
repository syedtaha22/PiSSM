import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import Models from '@/components/pages/models'
import * as api from '@/lib/api'

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

describe('Models', () => {
  it('renders the model list from listModels()', async () => {
    vi.spyOn(api, 'listModels').mockResolvedValue([SAMPLE_MODEL])

    render(<Models />)

    expect(screen.getByText('Models')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('dummy-mamba-tiny')).toBeInTheDocument())
  })

  it('shows an error message when listModels() fails', async () => {
    vi.spyOn(api, 'listModels').mockRejectedValue(new Error('orchestrator unreachable'))

    render(<Models />)

    await waitFor(() =>
      expect(screen.getByText('orchestrator unreachable')).toBeInTheDocument()
    )
  })
})
