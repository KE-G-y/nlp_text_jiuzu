import type {
  ClassificationCandidate,
  ClassificationResult,
} from '../types/classification'

// 通用对象类型，用于安全读取接口返回中的动态字段。
type Dictionary = Record<string, unknown>

// Mock 分类规则配置：用于本地预览或未配置接口时模拟分类结果。
interface MockProfile {
  keywords: string[]
  categoryName: string
  categoryCode: string
  categoryPath: string[]
  confidence: number
  reason: string
}

// 本地模拟分类规则库，通过商品标题关键词匹配对应类目。
const mockProfiles: MockProfile[] = [
  {
    keywords: ['iphone', '手机', '华为', '小米', 'oppo', 'vivo'],
    categoryName: '手机',
    categoryCode: '3C-MOBILE-PHONE',
    categoryPath: ['数码家电', '手机通讯', '手机'],
    confidence: 0.93,
    reason: '标题包含品牌或机型词，优先命中手机通讯类目。',
  },
  {
    keywords: ['耳机', '蓝牙', '降噪', '音箱'],
    categoryName: '影音配件',
    categoryCode: '3C-AUDIO-ACCESSORY',
    categoryPath: ['数码家电', '影音娱乐', '影音配件'],
    confidence: 0.89,
    reason: '标题包含音频设备关键词，归入影音相关类目。',
  },
  {
    keywords: ['连衣裙', '女装', '衬衫', '外套', '雪纺'],
    categoryName: '连衣裙',
    categoryCode: 'FASHION-DRESS',
    categoryPath: ['服饰鞋包', '女装', '连衣裙'],
    confidence: 0.91,
    reason: '标题包含女装款式词，匹配服饰鞋包下的女装类目。',
  },
  {
    keywords: ['空气炸锅', '电饭煲', '破壁机', '微波炉'],
    categoryName: '厨房电器',
    categoryCode: 'HOME-KITCHEN-APPLIANCE',
    categoryPath: ['家用电器', '厨房电器'],
    confidence: 0.9,
    reason: '标题包含厨房小家电关键词，匹配厨房电器类目。',
  },
  {
    keywords: ['积木', '玩具', '儿童', '益智'],
    categoryName: '益智玩具',
    categoryCode: 'TOY-EDUCATIONAL',
    categoryPath: ['母婴玩具', '玩具乐器', '益智玩具'],
    confidence: 0.87,
    reason: '标题包含儿童玩具和益智关键词，匹配玩具乐器类目。',
  },
  {
    keywords: ['洗面奶', '面霜', '精华', '口红', '防晒'],
    categoryName: '面部护肤',
    categoryCode: 'BEAUTY-SKINCARE',
    categoryPath: ['美妆个护', '面部护肤'],
    confidence: 0.88,
    reason: '标题包含美妆护肤词，匹配面部护肤类目。',
  },
]

// 商品标题分类入口函数：根据配置决定调用真实接口或使用本地 Mock。
export async function classifyProductTitle(
  title: string,
): Promise<ClassificationResult> {
  // 去除用户输入标题首尾空格，避免空白字符影响判断和请求。
  const normalizedTitle = title.trim()

  // 标题不能为空，否则直接抛出业务提示错误。
  if (!normalizedTitle) {
    throw new Error('请输入商品标题')
  }

  const endpoint = getEndpoint()

  // 未配置接口或显式开启 Mock 时，返回本地模拟分类结果。
  if (shouldUseMock(endpoint)) {
    await wait(420)
    return buildMockResult(normalizedTitle)
  }

  // 使用 AbortController 实现请求超时控制。
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), getTimeoutMs())

  try {
    // 调用后端分类接口，将商品标题以 JSON 格式提交。
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ title: normalizedTitle }),
      signal: controller.signal,
    })

    // HTTP 状态码非 2xx 时，视为接口请求失败。
    if (!response.ok) {
      throw new Error(`分类接口请求失败：HTTP ${response.status}`)
    }

    // 将不同格式的接口响应统一转换为前端需要的 ClassificationResult。
    const payload = (await response.json()) as unknown
    return normalizeClassificationResponse(payload, normalizedTitle)
  } catch (error) {
    // 请求被超时中断时，返回更明确的超时提示。
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('分类接口请求超时，请稍后重试')
    }

    // 保留原始 Error；非 Error 类型统一包装为请求失败。
    throw error instanceof Error ? error : new Error('分类接口请求失败')
  } finally {
    // 无论请求成功或失败，都清理定时器，避免资源泄漏。
    window.clearTimeout(timeoutId)
  }
}

// 读取分类接口地址环境变量。
function getEndpoint() {
  return String(import.meta.env.VITE_CLASSIFY_API_URL || '').trim()
}

// 读取接口超时时间环境变量，非法值时使用默认 12000ms。
function getTimeoutMs() {
  const value = Number(import.meta.env.VITE_CLASSIFY_TIMEOUT_MS || 12000)
  return Number.isFinite(value) && value > 0 ? value : 12000
}

// 判断是否使用本地 Mock：显式配置优先，其次根据接口地址是否为空判断。
function shouldUseMock(endpoint: string) {
  const value = String(import.meta.env.VITE_USE_MOCK || '').toLowerCase()

  if (['true', '1', 'yes'].includes(value)) {
    return true
  }

  if (['false', '0', 'no'].includes(value)) {
    return false
  }

  return endpoint.length === 0
}

// 兼容不同后端响应结构，将接口返回标准化为 ClassificationResult。
function normalizeClassificationResponse(
  payload: unknown,
  title: string,
): ClassificationResult {
  // 兼容常见响应格式：payload、payload.data、payload.result、payload.data.result。
  const root = asDictionary(payload) ?? {}
  const data = asDictionary(root.data) ?? asDictionary(root.result) ?? root
  const body = asDictionary(data.result) ?? data
  const categoryPath = normalizePath(
    body.categoryPath ?? body.category_path ?? body.path,
  )
  const categoryName =
    pickString(body, ['categoryName', 'category_name', 'category', 'label']) ||
    categoryPath.at(-1) ||
    '未识别分类'
  const confidence = pickNumber(body, ['confidence', 'score', 'probability'])
  const candidates = normalizeCandidates(
    body.candidates ?? body.topK ?? body.predictions,
    categoryPath,
    confidence,
  )

  return {
    title,
    categoryName,
    categoryCode: pickString(body, ['categoryCode', 'category_code', 'code']),
    categoryPath,
    confidence,
    candidates,
    reason: pickString(body, ['reason', 'explanation', 'message']),
    modelVersion: pickString(body, ['modelVersion', 'model_version', 'model']),
    requestId: pickString(body, ['requestId', 'request_id', 'traceId']),
    source: 'api',
    raw: payload,
  }
}

// 标准化候选分类列表，兼容字符串数组和对象数组两种候选结果。
function normalizeCandidates(
  value: unknown,
  fallbackPath: string[],
  fallbackConfidence?: number,
): ClassificationCandidate[] {
  // 没有候选列表时，使用主分类路径生成一个兜底候选项。
  if (!Array.isArray(value)) {
    if (fallbackPath.length === 0) {
      return []
    }

    return [
      {
        name: fallbackPath.at(-1) || '未识别分类',
        path: fallbackPath,
        confidence: fallbackConfidence,
      },
    ]
  }

  return value
    .map((item) => {
      // 候选项是字符串时，将其解析为分类路径。
      if (typeof item === 'string') {
        const path = normalizePath(item)
        return {
          name: path.at(-1) || item,
          path: path.length > 0 ? path : [item],
        }
      }

      const itemData = asDictionary(item)
      if (!itemData) {
        return null
      }

      // 候选项是对象时，读取类目名称、编码、路径和置信度。
      const path = normalizePath(
        itemData.categoryPath ?? itemData.category_path ?? itemData.path,
      )
      const name =
        pickString(itemData, ['name', 'categoryName', 'category', 'label']) ||
        path.at(-1) ||
        '候选分类'

      return {
        name,
        code: pickString(itemData, ['code', 'categoryCode', 'category_code']),
        path: path.length > 0 ? path : [name],
        confidence: pickNumber(itemData, ['confidence', 'score', 'probability']),
      }
    })
    .filter((item): item is ClassificationCandidate => item !== null)
}

// 根据本地关键词规则生成 Mock 分类结果。
function buildMockResult(title: string): ClassificationResult {
  const lowerTitle = title.toLowerCase()
  const profile =
    mockProfiles.find((item) =>
      item.keywords.some((keyword) => lowerTitle.includes(keyword)),
    ) ?? buildFallbackProfile()

  return {
    title,
    categoryName: profile.categoryName,
    categoryCode: profile.categoryCode,
    categoryPath: profile.categoryPath,
    confidence: profile.confidence,
    candidates: buildMockCandidates(profile),
    reason: profile.reason,
    modelVersion: 'mock-preview',
    requestId: `mock-${Date.now()}`,
    source: 'mock',
  }
}

// 构造 Mock 候选分类：当前命中类目排第一，再补充两个其他类目。
function buildMockCandidates(profile: MockProfile): ClassificationCandidate[] {
  const pool = mockProfiles
    .filter((item) => item.categoryCode !== profile.categoryCode)
    .slice(0, 2)

  return [profile, ...pool].map((item, index) => ({
    name: item.categoryName,
    code: item.categoryCode,
    path: item.categoryPath,
    confidence: Math.max(0.42, item.confidence - index * 0.12),
  }))
}

// 未命中任何 Mock 关键词时返回的兜底分类配置。
function buildFallbackProfile(): MockProfile {
  return {
    keywords: [],
    categoryName: '待确认类目',
    categoryCode: 'GENERAL-PENDING',
    categoryPath: ['通用商品', '待确认类目'],
    confidence: 0.61,
    reason: '未命中本地预览关键词，真实分类结果以接口返回为准。',
  }
}

// 将分类路径标准化为字符串数组，支持数组或用分隔符拼接的字符串。
function normalizePath(value: unknown) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean)
  }

  if (typeof value === 'string') {
    return value
      .split(/[>/|,，]+/)
      .map((item) => item.trim())
      .filter(Boolean)
  }

  return []
}

// 按候选字段名顺序读取第一个有效字符串。
function pickString(source: Dictionary, keys: string[]) {
  for (const key of keys) {
    const value = source[key]
    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return undefined
}

// 按候选字段名顺序读取第一个有效数字，并转换为 0 到 1 的置信度。
function pickNumber(source: Dictionary, keys: string[]) {
  for (const key of keys) {
    const value = source[key]
    if (typeof value === 'number' && Number.isFinite(value)) {
      return normalizeConfidence(value)
    }

    if (typeof value === 'string' && value.trim()) {
      const numberValue = Number(value)
      if (Number.isFinite(numberValue)) {
        return normalizeConfidence(numberValue)
      }
    }
  }

  return undefined
}

// 统一置信度范围：支持 0-1 小数和 0-100 百分制输入。
function normalizeConfidence(value: number) {
  if (value > 1) {
    return Math.max(0, Math.min(value / 100, 1))
  }

  return Math.max(0, Math.min(value, 1))
}

// 将 unknown 安全收窄为普通对象，方便读取动态字段。
function asDictionary(value: unknown): Dictionary | undefined {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Dictionary
  }

  return undefined
}

// 简单延迟函数，用于模拟接口请求耗时。
function wait(ms: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}
