<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  AlertCircle,
  BadgeCheck,
  ChartNoAxesColumn,
  LoaderCircle,
  Search,
  Tags,
  X,
} from '@lucide/vue'
import { classifyProductTitle } from './services/classification'
import type { ClassificationResult } from './types/classification'

const examples = [
  'Apple iPhone 15 Pro Max 256GB 黑色 原封国行',
  '法式碎花雪纺连衣裙女夏季收腰显瘦',
  '家用空气炸锅大容量智能无油电炸锅',
  '儿童益智拼装积木玩具生日礼物',
]

const title = ref('')
const loading = ref(false)
const errorMessage = ref('')
const result = ref<ClassificationResult | null>(null)

const trimmedTitle = computed(() => title.value.trim())
const canSubmit = computed(() => trimmedTitle.value.length > 1 && !loading.value)
const titleLength = computed(() => trimmedTitle.value.length)
const displayPath = computed(() => {
  if (!result.value) {
    return []
  }

  return result.value.categoryPath.length > 0
    ? result.value.categoryPath
    : [result.value.categoryName]
})
const confidencePercent = computed(() => formatConfidence(result.value?.confidence))
const confidenceBarWidth = computed(() => {
  const value = result.value?.confidence
  if (typeof value !== 'number') {
    return '0%'
  }

  return `${Math.round(Math.max(0, Math.min(value, 1)) * 100)}%`
})
const resultSourceText = computed(() => {
  if (!result.value) {
    return ''
  }

  return result.value.source === 'api' ? '接口结果' : '本地预览'
})

function formatConfidence(value?: number) {
  if (typeof value !== 'number') {
    return '待确认'
  }

  return `${Math.round(Math.max(0, Math.min(value, 1)) * 100)}%`
}

function useExample(sample: string) {
  title.value = sample
  errorMessage.value = ''
  result.value = null
}

function resetAll() {
  title.value = ''
  errorMessage.value = ''
  result.value = null
}

async function submitTitle() {
  if (!canSubmit.value) {
    return
  }

  loading.value = true
  errorMessage.value = ''
  result.value = null

  try {
    result.value = await classifyProductTitle(trimmedTitle.value)
  } catch (error) {
    errorMessage.value =
      error instanceof Error ? error.message : '分类失败，请稍后重试'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <main class="app-shell">
    <header class="topbar" aria-labelledby="page-title">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <Tags :size="24" />
        </div>
        <div>
          <p class="eyebrow">商品智能发布</p>
          <h1 id="page-title">商品标题分类</h1>
        </div>
      </div>

      <div class="version-badges" aria-label="前端技术栈">
        <span>Vue 3.5</span>
        <span>Vite 8</span>
        <span>TypeScript</span>
      </div>
    </header>

    <div class="workspace">
      <form class="panel input-panel" @submit.prevent="submitTitle">
        <div class="panel-head">
          <div>
            <p class="eyebrow">输入</p>
            <h2>商品标题</h2>
          </div>

          <button
            v-if="title || result || errorMessage"
            class="icon-button"
            type="button"
            title="清空"
            aria-label="清空"
            @click="resetAll"
          >
            <X :size="18" />
          </button>
        </div>

        <label class="sr-only" for="product-title">商品标题</label>
        <textarea
          id="product-title"
          v-model="title"
          :disabled="loading"
          maxlength="120"
          placeholder="例如：Apple iPhone 15 Pro Max 256GB 黑色 原封国行"
        />

        <div class="form-footer">
          <span class="counter">{{ titleLength }}/120</span>
          <button class="primary-button" type="submit" :disabled="!canSubmit">
            <LoaderCircle v-if="loading" class="spin" :size="18" />
            <Search v-else :size="18" />
            <span>{{ loading ? '分类中' : '开始分类' }}</span>
          </button>
        </div>

        <div class="examples" aria-label="示例标题">
          <button
            v-for="sample in examples"
            :key="sample"
            class="example-button"
            type="button"
            @click="useExample(sample)"
          >
            {{ sample }}
          </button>
        </div>
      </form>

      <section class="panel result-panel" aria-live="polite">
        <div class="panel-head">
          <div>
            <p class="eyebrow">输出</p>
            <h2>分类结果</h2>
          </div>
          <ChartNoAxesColumn :size="22" class="panel-icon" aria-hidden="true" />
        </div>

        <div v-if="loading" class="state">
          <LoaderCircle class="state-icon spin" :size="36" />
          <h3>正在识别类目</h3>
        </div>

        <div v-else-if="errorMessage" class="state state-error">
          <AlertCircle class="state-icon" :size="36" />
          <h3>分类失败</h3>
          <p>{{ errorMessage }}</p>
        </div>

        <div v-else-if="!result" class="state">
          <Search class="state-icon" :size="36" />
          <h3>暂无结果</h3>
        </div>

        <div v-else class="result-content">
          <div class="result-main">
            <div class="category-icon" aria-hidden="true">
              <BadgeCheck :size="28" />
            </div>
            <div>
              <p class="eyebrow">{{ resultSourceText }}</p>
              <h2>{{ result.categoryName }}</h2>
            </div>
            <span class="confidence-badge">{{ confidencePercent }}</span>
          </div>

          <div class="path-strip" aria-label="分类路径">
            <span v-for="node in displayPath" :key="node" class="path-node">
              {{ node }}
            </span>
          </div>

          <dl class="metadata">
            <div>
              <dt>类目编码</dt>
              <dd>{{ result.categoryCode || '待返回' }}</dd>
            </div>
            <div>
              <dt>模型版本</dt>
              <dd>{{ result.modelVersion || '待返回' }}</dd>
            </div>
            <div>
              <dt>请求编号</dt>
              <dd>{{ result.requestId || '待返回' }}</dd>
            </div>
          </dl>

          <div class="meter" aria-label="置信度">
            <span :style="{ width: confidenceBarWidth }"></span>
          </div>

          <div v-if="result.candidates.length" class="candidates">
            <div class="section-title">
              <h3>候选分类</h3>
              <span>{{ result.candidates.length }} 项</span>
            </div>
            <ul>
              <li
                v-for="candidate in result.candidates"
                :key="`${candidate.code || candidate.name}-${candidate.path.join('/')}`"
                class="candidate-row"
              >
                <div>
                  <strong>{{ candidate.name }}</strong>
                  <p>{{ candidate.path.join(' / ') }}</p>
                </div>
                <span>{{ formatConfidence(candidate.confidence) }}</span>
              </li>
            </ul>
          </div>

          <p v-if="result.reason" class="reason">{{ result.reason }}</p>

          <details v-if="result.raw" class="raw-response">
            <summary>原始返回</summary>
            <pre>{{ JSON.stringify(result.raw, null, 2) }}</pre>
          </details>
        </div>
      </section>
    </div>
  </main>
</template>
