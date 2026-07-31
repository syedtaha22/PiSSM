'use client'

import { useEffect, useRef, useState } from 'react'
import { Upload } from 'lucide-react'
import { listModels, submitModel, type ModelSummary } from '@/lib/api'

export default function Models() {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

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
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-light text-foreground">Models</h2>
        <div className="flex flex-col items-end gap-1">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-2 px-4 py-2 text-sm text-primary hover:text-primary/80 transition-colors border-b border-primary disabled:opacity-50"
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

      {error && <div className="text-sm text-destructive">{error}</div>}
      {!error && loading && (
        <div className="text-sm text-muted-foreground">Loading models...</div>
      )}
      {!error && !loading && models.length === 0 && (
        <div className="text-sm text-muted-foreground">No models registered yet.</div>
      )}

      <div className="space-y-4 divide-y divide-border">
        {models.map((model) => (
          <div key={model.name} className="py-4 space-y-2">
            <div className="flex items-baseline justify-between">
              <div>
                <h3 className="text-base font-medium text-foreground">{model.name}</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {model.arch} • {model.layers} layers • {model.checkpoint}
                </p>
              </div>
            </div>
            <div className="flex gap-8 text-xs text-muted-foreground pt-2">
              <div>Input: {model.input_type}</div>
              <div>State Dim: {model.state_dim}</div>
              <div>Hidden Dim: {model.hidden_dim}</div>
              <div>Tokenizer: {model.tokenizer}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
