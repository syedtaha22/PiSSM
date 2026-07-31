import { Zap } from 'lucide-react'

export default function Header() {
  return (
    <header className="border-b border-border px-8 py-5 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <Zap size={20} className="text-primary" />
        <h1 className="text-lg font-semibold text-foreground">PiSSM</h1>
        <span className="text-xs text-muted-foreground">distributed inference</span>
      </div>
      <div className="flex items-center gap-3 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
          <span className="text-muted-foreground">Active</span>
        </div>
      </div>
    </header>
  )
}
