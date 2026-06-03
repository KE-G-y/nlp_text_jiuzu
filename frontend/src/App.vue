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
import { productCategories } from './constants/categories'
import { classifyProductTitle } from './services/classification'
import type { ClassificationResult } from './types/classification'

const examples = [
  '美的电热水壶304不锈钢MK-SP50Colour201',
  '好奇心钻装纸尿裤L40片9-14kg',
  '潘婷丝质顺滑洗发露750ml',
  '风味坐标 三鲜狮子头200g/2只',
]

const modelOptions = [
  { label: 'Fasttext字符模型', value: 'fasttext_char' },
  { label: 'Fasttext词模型', value: 'fasttext_word' },
  { label: 'bert模型', value: 'bert' },
]

const title = ref('')
const selectedModel = ref(modelOptions[0].value)
const loading = ref(false)
const errorMessage = ref('')
const result = ref<ClassificationResult | null>(null)

const trimmedTitle = computed(() => title.value.trim())
const canSubmit = computed(() => trimmedTitle.value.length > 1 && !loading.value)
const titleLength = computed(() => trimmedTitle.value.length)
const resultSourceText = computed(() => {
  if (!result.value) {
    return ''
  }

  return result.value.source === 'api' ? '接口结果' : '本地预览'
})
const resultStatusText = computed(() => {
  if (!result.value) {
    return ''
  }

  return result.value.isKnownCategory ? '已命中' : '未收录'
})

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

function isSelectedCategory(category: string) {
  return result.value?.categoryName === category
}

function getModelLabel(value: string) {
  return modelOptions.find((item) => item.value === value)?.label ?? value
}

async function submitTitle() {
  if (!canSubmit.value) {
    return
  }

  loading.value = true
  errorMessage.value = ''
  result.value = null

  try {
    result.value = await classifyProductTitle(trimmedTitle.value, selectedModel.value)
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
        <span>30 类</span>
        <span>单标签</span>
        <span>Vue 3.5</span>
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

        <label class="field-label" for="model-select">模型选择</label>
        <select id="model-select" v-model="selectedModel" :disabled="loading">
          <option
            v-for="model in modelOptions"
            :key="model.value"
            :value="model.value"
          >
            {{ model.label }}
          </option>
        </select>

        <label class="sr-only" for="product-title">商品标题</label>
        <textarea
          id="product-title"
          v-model="title"
          :disabled="loading"
          maxlength="120"
          placeholder="例如：美的电热水壶304不锈钢MK-SP50Colour201"
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

        <div class="result-body">
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

          <div v-else class="single-result">
            <div
              class="category-icon"
              :class="{ 'category-icon-warning': !result.isKnownCategory }"
              aria-hidden="true"
            >
              <BadgeCheck :size="28" />
            </div>
            <div class="single-result-main">
              <p class="eyebrow">{{ resultSourceText }}</p>
              <h2>{{ result.categoryName }}</h2>
              <p class="title-preview">{{ result.title }}</p>
              <div class="result-meta">
                <span>模型：{{ getModelLabel(result.modelName) }}</span>
                <span>耗时：{{ result.durationMs }} ms</span>
              </div>
            </div>
            <span
              class="result-status"
              :class="{ 'result-status-warning': !result.isKnownCategory }"
            >
              {{ resultStatusText }}
            </span>
          </div>

          <section class="category-section" aria-labelledby="category-list-title">
            <div class="section-title">
              <h3 id="category-list-title">分类范围</h3>
              <span>{{ productCategories.length }} 类</span>
            </div>
            <div class="category-grid" role="list">
              <span
                v-for="(category, index) in productCategories"
                :key="category"
                class="category-chip"
                :class="{ 'category-chip-active': isSelectedCategory(category) }"
                role="listitem"
              >
                <span>{{ String(index + 1).padStart(2, '0') }}</span>
                {{ category }}
              </span>
            </div>
          </section>

          <details v-if="result?.raw" class="raw-response">
            <summary>原始返回</summary>
            <pre>{{ JSON.stringify(result.raw, null, 2) }}</pre>
          </details>
        </div>
      </section>
    </div>
  </main>
</template>
