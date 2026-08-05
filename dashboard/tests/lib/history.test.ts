import { describe, it, expect, afterEach } from 'vitest'
import { appendInferenceLog, getInferenceLog } from '@/lib/history'

afterEach(() => {
  window.localStorage.clear()
})

describe('getInferenceLog', () => {
  it('returns an empty array when nothing has been logged', () => {
    expect(getInferenceLog()).toEqual([])
  })
})

describe('appendInferenceLog', () => {
  it('persists an entry that getInferenceLog then returns', () => {
    appendInferenceLog({ timestamp: 1, modelName: 'dummy-mamba-tiny', latencyMs: 120, numNodes: 1 })

    expect(getInferenceLog()).toEqual([
      { timestamp: 1, modelName: 'dummy-mamba-tiny', latencyMs: 120, numNodes: 1 },
    ])
  })

  it('appends rather than overwriting previous entries', () => {
    appendInferenceLog({ timestamp: 1, modelName: 'dummy-mamba-tiny', latencyMs: 120, numNodes: 1 })
    appendInferenceLog({ timestamp: 2, modelName: 'dummy-mamba-tiny', latencyMs: 340, numNodes: 1 })

    expect(getInferenceLog()).toHaveLength(2)
  })

  it('persists numTokens when provided', () => {
    appendInferenceLog({
      timestamp: 1,
      modelName: 'dummy-mamba-tiny',
      latencyMs: 120,
      numNodes: 1,
      numTokens: 20,
    })

    expect(getInferenceLog()).toEqual([
      { timestamp: 1, modelName: 'dummy-mamba-tiny', latencyMs: 120, numNodes: 1, numTokens: 20 },
    ])
  })

  it('caps the log at 50 entries, dropping the oldest first', () => {
    for (let i = 0; i < 55; i++) {
      appendInferenceLog({ timestamp: i, modelName: 'dummy-mamba-tiny', latencyMs: i, numNodes: 1 })
    }

    const log = getInferenceLog()
    expect(log).toHaveLength(50)
    expect(log[0].timestamp).toBe(5)
    expect(log[log.length - 1].timestamp).toBe(54)
  })
})
