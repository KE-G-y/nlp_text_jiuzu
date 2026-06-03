# 商品标题分类前端

基于 Vue 3 + Vite + TypeScript 的商品标题分类页面。用户输入商品标题后，前端会调用后端分类接口并展示类目、置信度、候选分类和原始返回。未配置接口时会自动使用本地 mock 结果，方便先联调页面。

## 技术栈

- Vue `^3.5.35`
- Vite `^8.0.16`
- TypeScript `~6.0.2`
- @lucide/vue `^1.17.0`

## 目录结构

```text
frontend
├── src
│   ├── App.vue                     # 页面入口、状态和交互
│   ├── main.ts                     # Vue 应用挂载
│   ├── style.css                   # 页面样式
│   ├── services
│   │   └── classification.ts       # 分类接口调用和返回值归一化
│   └── types
│       └── classification.ts       # 分类结果类型定义
├── .env.example                    # 环境变量模板
├── package.json
└── vite.config.ts
```

## 本地运行

```bash
npm install
npm run dev
```

打开终端输出的本地地址即可访问页面，通常是 `http://localhost:5173/`。

## 后端接口配置

复制环境变量模板：

```bash
cp .env.example .env
```

配置真实接口：

```env
VITE_CLASSIFY_API_URL=/api/classify
VITE_USE_MOCK=false
VITE_CLASSIFY_TIMEOUT_MS=12000
```

如果 `VITE_CLASSIFY_API_URL` 为空，且没有显式设置 `VITE_USE_MOCK=false`，前端会使用本地 mock 结果。

## 接口契约

请求方式：`POST`

请求体：

```json
{
  "title": "Apple iPhone 15 Pro Max 256GB 黑色 原封国行"
}
```

推荐响应体：

```json
{
  "requestId": "cls-20260603-0001",
  "modelVersion": "product-classifier-v1",
  "categoryName": "手机",
  "categoryCode": "3C-MOBILE-PHONE",
  "categoryPath": ["数码家电", "手机通讯", "手机"],
  "confidence": 0.93,
  "reason": "标题包含品牌和机型词，匹配手机类目。",
  "candidates": [
    {
      "name": "手机",
      "code": "3C-MOBILE-PHONE",
      "path": ["数码家电", "手机通讯", "手机"],
      "confidence": 0.93
    }
  ]
}
```

前端也兼容部分常见字段别名，例如 `category_name`、`category`、`score`、`probability`、`topK`、`predictions`。

## 构建

```bash
npm run build
```

构建产物会输出到 `dist` 目录。
