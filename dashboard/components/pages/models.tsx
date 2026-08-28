'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { Upload } from 'lucide-react'
import { listModels, submitModel, type ModelSummary } from '@/lib/api'
import { SearchInput } from '@/components/ui/search-input'

export default function Models() {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [query, setQuery] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const visibleModels = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return models
    return models.filter(
      (m) => m.name.toLowerCase().includes(q) || m.arch.toLowerCase().includes(q)
    )
  }, [models, query])

  useEffect(() => {
    let cancelled = false

    listModels()
      .then((result) => {
        if (!cancelled) {
          setModels(result)
          setError(null)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load models')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

  async function refreshAfterUpload() {
    try {
      const result = await listModels()
      setModels(result)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models')
    }
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    setUploading(true)
    setUploadError(null)
    try {
      const manifestYaml = await file.text()
      await submitModel(manifestYaml)
      await refreshAfterUpload()
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Failed to submit manifest')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="mx-auto w-full max-w-[1120px] space-y-8 p-8">
      <div className="flex items-center justify-between">
        <h2 className="font-display text-2xl font-semibold tracking-tight text-foreground">
          Models
        </h2>
        <div className="flex flex-col items-end gap-1">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-2 rounded-sm px-4 py-2 font-sans text-sm font-medium text-primary hover:text-primary/80 transition-colors border-b border-primary disabled:opacity-50 cursor-pointer"
          >
            <Upload size={16} />
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".yaml,.yml"
            onChange={handleFileChange}
            className="hidden"
          />
          {uploadError && <div className="text-xs text-destructive">{uploadError}</div>}
        </div>
      </div>

      {!error && !loading && models.length > 0 && (
        <SearchInput
          placeholder="Search models..."
          value={query}
          onChange={setQuery}
          className="max-w-sm"
        />
      )}

      {error && <div className="text-sm text-destructive">{error}</div>}
      {!error && loading && (
        <div className="text-sm text-muted-foreground">Loading models...</div>
      )}
      {!error && !loading && models.length === 0 && (
        <div className="text-sm text-muted-foreground">No models registered yet.</div>
      )}
      {!error && !loading && models.length > 0 && visibleModels.length === 0 && (
        <div className="text-sm text-muted-foreground">No models match &ldquo;{query}&rdquo;.</div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {visibleModels.map((model) => (
          <div
            key={model.name}
            className="rounded-md border border-border bg-card p-6 transition-colors hover:border-border-strong"
          >
            <h3 className="font-display text-base font-semibold text-foreground">
              {model.name}
            </h3>
            <p className="mt-1 font-sans text-sm text-muted-foreground">
              {model.arch} • {model.layers} layers • {model.checkpoint}
            </p>
            <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1.5 border-t border-border pt-3 font-mono text-xs text-muted-foreground">
              <div className="whitespace-nowrap">Input: {model.input_type}</div>
              <div className="whitespace-nowrap">State dim: {model.state_dim}</div>
              <div className="whitespace-nowrap">Hidden dim: {model.hidden_dim}</div>
              <div className="whitespace-nowrap">Tokenizer: {model.tokenizer}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
