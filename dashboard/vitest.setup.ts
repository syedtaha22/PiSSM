import '@testing-library/jest-dom/vitest'

// jsdom doesn't implement ResizeObserver, which reactflow relies on.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}
