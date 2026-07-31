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
    appendInferenceLog({ timestamp: 1, modelName: 'mamba-130m', latencyMs: 120, numNodes: 1 })

    expect(getInferenceLog()).toEqual([
      { timestamp: 1, modelName: 'mamba-130m', latencyMs: 120, numNodes: 1 },
    ])
  })

  it('appends rather than overwriting previous entries', () => {
    appendInferenceLog({ timestamp: 1, modelName: 'mamba-130m', latencyMs: 120, numNodes: 1 })
    appendInferenceLog({ timestamp: 2, modelName: 'mamba-130m', latencyMs: 340, numNodes: 1 })

    expect(getInferenceLog()).toHaveLength(2)
  })

  it('caps the log at 50 entries, dropping the oldest first', () => {
    for (let i = 0; i < 55; i++) {
      appendInferenceLog({ timestamp: i, modelName: 'mamba-130m', latencyMs: i, numNodes: 1 })
    }

    const log = getInferenceLog()
    expect(log).toHaveLength(50)
    expect(log[0].timestamp).toBe(5)
    expect(log[log.length - 1].timestamp).toBe(54)
  })
})
