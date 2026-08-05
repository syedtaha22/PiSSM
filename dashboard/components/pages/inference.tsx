'use client'

import { useEffect, useRef, useState } from 'react'
import { Send, Loader2, RefreshCw } from 'lucide-react'
import {
  listModels,
  loadModel,
  redistributeModel,
  getModelStatus,
  runInferenceStream,
  type ModelSummary,
  type ModelLoadStatus,
} from '@/lib/api'
import { appendInferenceLog } from '@/lib/history'

const STATUS_POLL_INTERVAL_MS = 500
const MODELS_POLL_INTERVAL_MS = 3000

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
}

export default function Inference() {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState('')
  const [modelStatus, setModelStatus] = useState<ModelLoadStatus>('not_loaded')
  const [modelStatusError, setModelStatusError] = useState<string | null>(null)
  const [numNodes, setNumNodes] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [metrics, setMetrics] = useState({ latency: '-', peakMemory: '-', nodes: '-', tokensPerSec: '-' })
  const [redistributing, setRedistributing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // Which endpoint the load-and-poll effect below should call: 'load' on
  // a fresh model selection, 'redistribute' when the button below is
  // clicked. A ref (not state) since it's read once when the effect
  // fires, not something the render itself depends on.
  const pendingActionRef = useRef<'load' | 'redistribute'>('load')
  const [loadTrigger, setLoadTrigger] = useState(0)

  // Poll the registered models list rather than fetching it once - pages
  // stay mounted for the whole session now (see app/page.tsx), so a model
  // uploaded on the Models tab after this one first mounted would
  // otherwise never appear here without a full page reload. Deliberately
  // does NOT auto-select (and therefore does not auto-load) the first
  // model - loading a model is an explicit, resource-committing action
  // the user takes from the dropdown, not something that should happen
  // just from visiting this tab or from a background refresh.
  useEffect(() => {
    let cancelled = false

    function poll() {
      listModels()
        .then((result) => {
          if (cancelled) return
          setModels(result)
          setModelsError(null)
        })
        .catch((err) => {
          if (!cancelled) {
            setModelsError(err instanceof Error ? err.message : 'Failed to load models')
          }
        })
    }

    poll()
    const interval = setInterval(poll, MODELS_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  // Selecting a model preloads it: kick off the load, then poll status
  // until it's ready (or errors) so the send button unlocks only once
  // the pipeline is actually resident in worker RAM. Also re-runs when
  // the Redistribute button bumps loadTrigger, calling redistributeModel
  // instead of loadModel.
  useEffect(() => {
    if (!selectedModel) return

    let cancelled = false
    const action = pendingActionRef.current === 'redistribute' ? redistributeModel : loadModel

    async function checkStatus() {
      try {
        const status = await getModelStatus(selectedModel)
        if (cancelled) return
        setModelStatus(status.status)
        setModelStatusError(status.error)
        setNumNodes(status.num_nodes)
        // Redistributing stays true (spinner + banner wording) for the
        // whole background reload, not just the initial POST - only
        // clear it once the poll actually observes a terminal status.
        if (status.status === 'ready' || status.status === 'error') {
          setRedistributing(false)
        }
      } catch (err) {
        if (!cancelled) {
          setModelStatus('error')
          setModelStatusError(err instanceof Error ? err.message : 'Failed to check status')
          setRedistributing(false)
        }
      }
    }

    async function start() {
      setModelStatus('loading')
      setModelStatusError(null)
      try {
        await action(selectedModel)
      } catch {
        /* checkStatus below reports the resulting error state */
      }
      await checkStatus()
    }

    start()
    const interval = setInterval(checkStatus, STATUS_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [selectedModel, loadTrigger])

  function handleSelectModel(name: string) {
    pendingActionRef.current = 'load'
    setSelectedModel(name)
  }

  function handleRedistribute() {
    if (redistributing || modelStatus === 'loading' || !selectedModel) return
    pendingActionRef.current = 'redistribute'
    setRedistributing(true)
    setLoadTrigger((t) => t + 1)
  }

  const modelType = models.find((m) => m.name === selectedModel)?.input_type ?? 'text'
  const modelReady = modelStatus === 'ready'
  const canSend = modelReady && !sending && input.trim().length > 0
  const canRedistribute =
    !redistributing && (modelStatus === 'ready' || modelStatus === 'error') && !!selectedModel

  const hasSentRef = useRef(false)

  // Re-focus the prompt input once sending finishes - it gets disabled
  // mid-request, and a disabled input drops browser focus, which
  // otherwise leaves the user having to re-click it for every prompt.
  useEffect(() => {
    if (!sending && hasSentRef.current) {
      inputRef.current?.focus()
    }
  }, [sending])

  // Keep the transcript scrolled to the newest message - including
  // mid-stream, since appendToken() updates `messages` on every token
  // and the user shouldn't have to manually scroll down as text grows.
  useEffect(() => {
    const el = chatContainerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const handleSend = async () => {
    if (!canSend) return

    hasSentRef.current = true
    const userMessage: Message = {
      id: messages.length + 1,
      role: 'user',
      content: input,
    }
    const assistantId = messages.length + 2
    const assistantMessage: Message = { id: assistantId, role: 'assistant', content: '' }

    setMessages((prev) => [...prev, userMessage, assistantMessage])
    setInput('')
    setSending(true)

    function appendToken(token: string) {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + token } : m))
      )
    }

    try {
      const result = await runInferenceStream(selectedModel, input, appendToken)
      const tokensPerSec =
        result.latency_ms > 0 ? result.num_tokens / (result.latency_ms / 1000) : 0
      setMetrics({
        latency: `${result.latency_ms.toFixed(0)}ms`,
        peakMemory: `${Math.max(...result.peak_memory_mb, 0)}MB`,
        nodes: String(result.num_nodes),
        tokensPerSec: `${tokensPerSec.toFixed(1)} tok/s`,
      })
      appendInferenceLog({
        timestamp: Date.now(),
        modelName: selectedModel,
        latencyMs: result.latency_ms,
        numNodes: result.num_nodes,
        numTokens: result.num_tokens,
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Inference failed'
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: `Error: ${message}` } : m
        )
      )
    } finally {
      setSending(false)
    }
  }

  function statusMessage(): string | null {
    if (modelsError) return `Could not reach orchestrator: ${modelsError}`
    if (models.length === 0) return 'No models registered. Upload a manifest on the Models page.'
    if (modelStatus === 'loading') {
      const verb = redistributing ? 'Redistributing' : 'Loading'
      return numNodes
        ? `${verb} model onto ${numNodes} node${numNodes === 1 ? '' : 's'}...`
        : `${verb} model...`
    }
    if (modelStatus === 'error') return `Failed to load model: ${modelStatusError}`
    return null
  }

  const banner = statusMessage()

  return (
    <div className="p-8 h-full flex flex-col">
      {/* Model Selection */}
      <div className="mb-8 space-y-2">
        <label htmlFor="model-select" className="text-sm font-medium text-foreground">
          Model
        </label>
        <div className="flex gap-2">
          <select
            id="model-select"
            value={selectedModel}
            onChange={(e) => handleSelectModel(e.target.value)}
            disabled={models.length === 0}
            className="flex-1 px-3 py-2 border border-border rounded bg-background text-foreground focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
          >
            {models.length === 0 ? (
              <option value="">No models registered</option>
            ) : (
              <option value="">Select a model...</option>
            )}
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
          <button
            onClick={handleRedistribute}
            disabled={!canRedistribute}
            title="Re-dispatch this model across the currently available nodes - use this after a node joins or leaves the cluster"
            className="px-3 py-2 border border-border rounded text-sm text-foreground hover:bg-muted disabled:opacity-50 transition-colors flex items-center gap-2"
          >
            <RefreshCw size={14} className={redistributing ? 'animate-spin' : undefined} />
            Redistribute
          </button>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground pt-1">
          <span>
            Type: <span className="font-mono">{modelType}</span>
          </span>
          {modelReady && (
            <span className="flex items-center gap-1 text-primary">
              <span className="w-1.5 h-1.5 rounded-full bg-primary" />
              Ready{numNodes ? ` on ${numNodes} node${numNodes === 1 ? '' : 's'}` : ''}
            </span>
          )}
        </div>
        {banner && (
          <div
            className={`flex items-center gap-2 text-xs rounded px-3 py-2 ${
              modelStatus === 'error' || modelsError
                ? 'bg-destructive/10 text-destructive'
                : 'bg-muted text-muted-foreground'
            }`}
          >
            {modelStatus === 'loading' && <Loader2 size={12} className="animate-spin" />}
            {banner}
          </div>
        )}
      </div>

      {/* Chat/Timeseries Area */}
      <div
        ref={chatContainerRef}
        className="flex-1 border-t border-border pt-6 pr-4 mb-6 overflow-y-auto overflow-x-hidden"
      >
        {modelType === 'text' ? (
          <div className="space-y-4">
            {messages.map((msg, i) => {
              const isStreamingPlaceholder =
                sending && msg.role === 'assistant' && msg.content === '' && i === messages.length - 1
              return (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xl px-4 py-2 rounded text-sm break-words whitespace-pre-wrap ${
                      msg.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-border text-foreground'
                    }`}
                  >
                    {isStreamingPlaceholder ? (
                      <span className="flex items-center gap-2 text-muted-foreground">
                        <Loader2 size={14} className="animate-spin" />
                        Generating...
                      </span>
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="w-full h-64 border border-border rounded bg-background/50 flex items-center justify-center">
            <div className="text-center text-muted-foreground text-sm">
              <div className="space-y-2">
                <div>Time Series Chart</div>
                <div className="text-xs">Charts will render timeseries model outputs here</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="space-y-4 border-t border-border pt-4">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                handleSend()
              }
            }}
            placeholder={
              !modelReady
                ? 'Waiting for model to load...'
                : modelType === 'text'
                  ? 'Enter prompt...'
                  : 'Enter data...'
            }
            className="flex-1 px-3 py-2 border border-border rounded bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary text-sm disabled:opacity-50"
            disabled={sending || !modelReady}
          />
          <button
            onClick={handleSend}
            disabled={!canSend}
            aria-label="Send"
            title={!modelReady ? 'Waiting for model to finish loading' : undefined}
            className="px-4 py-2 bg-primary text-primary-foreground rounded text-sm hover:bg-primary/90 disabled:opacity-50 transition-colors flex items-center gap-2"
          >
            <Send size={16} />
          </button>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-4 pt-4 border-t border-border">
          <div>
            <div className="text-xs text-muted-foreground">Latency</div>
            <div className="text-sm font-mono text-foreground">{metrics.latency}</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Tokens/sec</div>
            <div className="text-sm font-mono text-foreground">{metrics.tokensPerSec}</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Peak Memory</div>
            <div className="text-sm font-mono text-foreground">{metrics.peakMemory}</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Nodes Used</div>
            <div className="text-sm font-mono text-foreground">{metrics.nodes}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
