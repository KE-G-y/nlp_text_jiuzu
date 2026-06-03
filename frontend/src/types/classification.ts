export type ClassificationSource = 'api' | 'mock'

export interface ClassificationResult {
  title: string
  categoryName: string
  isKnownCategory: boolean
  source: ClassificationSource
  raw?: unknown
}
