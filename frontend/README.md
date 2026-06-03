# 商品标题分类前端

基于 Vue 3 + Vite + TypeScript 的商品标题分类页面。用户输入商品标题后，前端调用后端接口获取单标签分类结果，并在 30 个固定分类中高亮命中项。未配置接口时会自动使用本地 mock 结果，方便先联调页面。

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
│   ├── constants
│   │   └── categories.ts           # 30 个商品分类
│   ├── services
│   │   └── classification.ts       # 单标签分类接口调用和返回值归一化
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
  "title": "美的电热水壶304不锈钢MK-SP50Colour201"
}
```

推荐响应体：

```json
{
  "label": "家电"
}
```

前端也兼容以下单标签返回形式：

```json
"家电"
```

```json
{
  "data": "家电"
}
```

```json
{
  "result": {
    "category": "家电"
  }
}
```

兼容字段包括 `label`、`category`、`categoryName`、`category_name`、`class`、`className`、`prediction`、`predicted_label`、`predictedLabel`。

## 30 个分类

```text
3C数码、个护、书籍、乳品、休闲食品、健康、健康食品、办公、宠物、家居、家电、服饰内衣、母婴、水产、水果、汽车用品、清洁、玩具、礼品、粮油速食、美妆、肉禽蛋、蔬菜、运动、酒饮冲调、钟表配饰、鞋靴箱包、餐饮、香烟、鲜花绿植
```

## 构建

```bash
npm run build
```

构建产物会输出到 `dist` 目录。
