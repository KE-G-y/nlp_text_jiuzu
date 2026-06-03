import { productCategories } from '../constants/categories'
import type { ClassificationResult } from '../types/classification'

type Dictionary = Record<string, unknown>

interface MockRule {
  categoryName: string
  keywords: string[]
}

const mockRules: MockRule[] = [
  {
    categoryName: '3C数码',
    keywords: ['iphone', 'ipad', '手机', '电脑', '耳机', '充电', '数码'],
  },
  {
    categoryName: '个护',
    keywords: ['洗发', '牙膏', '牙刷', '卫生巾', '沐浴露', '纸尿裤'],
  },
  {
    categoryName: '书籍',
    keywords: ['图书', '绘本', '小说', '教材', '书籍'],
  },
  {
    categoryName: '乳品',
    keywords: ['牛奶', '酸奶', '奶酪', '乳酸菌'],
  },
  {
    categoryName: '休闲食品',
    keywords: ['薯片', '饼干', '巧克力', '零食', '糖果'],
  },
  {
    categoryName: '健康',
    keywords: ['口罩', '创可贴', '酒精', '棉签', '体温计'],
  },
  {
    categoryName: '健康食品',
    keywords: ['维生素', '蛋白粉', '燕窝', '虫草', '枸杞'],
  },
  {
    categoryName: '办公',
    keywords: ['中性笔', '文件夹', '打印纸', '笔记本', '文具'],
  },
  {
    categoryName: '宠物',
    keywords: ['猫粮', '狗粮', '宠物', '猫砂'],
  },
  {
    categoryName: '家居',
    keywords: ['餐具', '收纳', '水杯', '拖把', '毛巾', '被子'],
  },
  {
    categoryName: '家电',
    keywords: ['电饭煲', '空气炸锅', '电热水壶', '料理机', '吹风机'],
  },
  {
    categoryName: '服饰内衣',
    keywords: ['连衣裙', '内衣', 't恤', '外套', '套装', '衬衫'],
  },
  {
    categoryName: '母婴',
    keywords: ['婴儿', '奶粉', '纸尿裤', '儿童', '宝宝'],
  },
  {
    categoryName: '水产',
    keywords: ['虾', '鱼', '蟹', '贝', '鱿鱼', '水产'],
  },
  {
    categoryName: '水果',
    keywords: ['苹果', '梨', '橙', '桃', '葡萄', '水果'],
  },
  {
    categoryName: '汽车用品',
    keywords: ['汽车', '车载', '雨刷', '车用'],
  },
  {
    categoryName: '清洁',
    keywords: ['洗衣液', '清洁剂', '洗洁精', '垃圾袋', '消毒'],
  },
  {
    categoryName: '玩具',
    keywords: ['玩具', '积木', '遥控车', '篮球', '彩泥'],
  },
  {
    categoryName: '礼品',
    keywords: ['礼盒', '礼品', '贺卡', '节日'],
  },
  {
    categoryName: '粮油速食',
    keywords: ['大米', '食用油', '面条', '水饺', '火锅底料', '速食'],
  },
  {
    categoryName: '美妆',
    keywords: ['面霜', '口红', '面膜', '精华', '防晒'],
  },
  {
    categoryName: '肉禽蛋',
    keywords: ['猪肉', '牛肉', '鸡蛋', '鸡翅', '羊肉'],
  },
  {
    categoryName: '蔬菜',
    keywords: ['土豆', '玉米', '生菜', '蔬菜', '豆腐'],
  },
  {
    categoryName: '运动',
    keywords: ['运动', '瑜伽', '健身', '篮球', '跑步'],
  },
  {
    categoryName: '酒饮冲调',
    keywords: ['啤酒', '白酒', '饮料', '咖啡', '茶', '可乐'],
  },
  {
    categoryName: '钟表配饰',
    keywords: ['手表', '耳钉', '项链', '配饰'],
  },
  {
    categoryName: '鞋靴箱包',
    keywords: ['拖鞋', '运动鞋', '箱包', '行李箱', '书包'],
  },
  {
    categoryName: '餐饮',
    keywords: ['卤', '熟食', '便当', '丸子', '披萨'],
  },
  {
    categoryName: '香烟',
    keywords: ['香烟', '红塔山', '中华'],
  },
  {
    categoryName: '鲜花绿植',
    keywords: ['鲜花', '绿植', '花束', '银柳'],
  },
]

export async function classifyProductTitle(
  title: string,
  modelName: string,
): Promise<ClassificationResult> {
  const normalizedTitle = title.trim()

  if (!normalizedTitle) {
    throw new Error('请输入商品标题')
  }

  const endpoint = getEndpoint()
  const startedAt = performance.now()

  if (shouldUseMock(endpoint)) {
    await wait(360)
    return createResult(
      normalizedTitle,
      matchMockCategory(normalizedTitle),
      'mock',
      modelName,
      getDurationMs(startedAt),
    )
  }

  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), getTimeoutMs())

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: normalizedTitle, model: modelName }),
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new Error(`分类接口请求失败：HTTP ${response.status}`)
    }

    const payload = (await response.json()) as unknown
    const categoryName = normalizeSingleLabel(payload)

    if (!categoryName) {
      throw new Error('分类接口未返回有效标签')
    }

    return createResult(
      normalizedTitle,
      categoryName,
      'api',
      modelName,
      getDurationMs(startedAt),
      payload,
    )
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('分类接口请求超时，请稍后重试')
    }

    throw error instanceof Error ? error : new Error('分类接口请求失败')
  } finally {
    window.clearTimeout(timeoutId)
  }
}

function getEndpoint() {
  return String(import.meta.env.VITE_CLASSIFY_API_URL || '/api/predict').trim()
}

function getTimeoutMs() {
  const value = Number(import.meta.env.VITE_CLASSIFY_TIMEOUT_MS || 12000)
  return Number.isFinite(value) && value > 0 ? value : 12000
}

function getDurationMs(startedAt: number) {
  return Math.round(performance.now() - startedAt)
}

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

function createResult(
  title: string,
  categoryName: string,
  source: ClassificationResult['source'],
  modelName: string,
  durationMs: number,
  raw?: unknown,
): ClassificationResult {
  return {
    title,
    categoryName,
    isKnownCategory: productCategories.includes(
      categoryName as (typeof productCategories)[number],
    ),
    source,
    modelName,
    durationMs,
    raw,
  }
}

function matchMockCategory(title: string) {
  const normalizedTitle = title.toLowerCase()
  const rule = mockRules.find((item) =>
    item.keywords.some((keyword) => normalizedTitle.includes(keyword)),
  )

  return rule?.categoryName ?? '粮油速食'
}

function normalizeSingleLabel(payload: unknown): string {
  if (typeof payload === 'string') {
    return cleanLabel(payload)
  }

  if (Array.isArray(payload)) {
    return normalizeSingleLabel(payload[0])
  }

  const root = asDictionary(payload)
  if (!root) {
    return ''
  }

  const directLabel = pickString(root, [
    'label',
    'category',
    'categoryName',
    'category_name',
    'class',
    'className',
    'prediction',
    'pred_class',
    'predicted_label',
    'predictedLabel',
  ])

  if (directLabel) {
    return directLabel
  }

  for (const key of ['data', 'result', 'output']) {
    const nested = root[key]
    const nestedLabel = normalizeSingleLabel(nested)
    if (nestedLabel) {
      return nestedLabel
    }
  }

  return ''
}

function cleanLabel(value: string) {
  return value.trim().replace(/^["']|["']$/g, '')
}

function pickString(source: Dictionary, keys: string[]) {
  for (const key of keys) {
    const value = source[key]
    if (typeof value === 'string' && value.trim()) {
      return cleanLabel(value)
    }
  }

  return ''
}

function asDictionary(value: unknown): Dictionary | undefined {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Dictionary
  }

  return undefined
}

function wait(ms: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}
