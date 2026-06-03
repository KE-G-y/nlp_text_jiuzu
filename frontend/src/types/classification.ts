export type ClassificationSource = 'api' | 'mock'

export interface ClassificationCandidate {
  name: string
  code?: string
  path: string[]
  confidence?: number
}

export interface ClassificationResult {
  title: string
  categoryName: string
  categoryCode?: string
  categoryPath: string[]
  confidence?: number
  candidates: ClassificationCandidate[]
  reason?: string
  modelVersion?: string
  requestId?: string
  source: ClassificationSource
  raw?: unknown
}
