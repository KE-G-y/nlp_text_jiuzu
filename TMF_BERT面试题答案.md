# TMF 项目 BERT 相关面试题答案

## 1. BERT 数据预处理的思路及实现细节

我的项目是一个新闻文本 10 分类任务，原始数据格式是 `text\tlabel`，类别来自 `class.txt`，包括 `finance、realty、stocks、education、science、society、politics、sports、game、entertainment`。

数据预处理分三步：

1. 读取原始数据  
   在 `load_raw_data()` 中逐行读取 `train.txt、dev3.txt、test.txt`，去掉空行，再用 `\t` 切分成文本和标签，最终得到 `[(text, label), ...]` 的列表结构，标签转成 `int`。

2. 构造 BERT 输入  
   在 `collate_fn()` 中按 batch 取出 `texts` 和 `labels`，调用 `BertTokenizer` 对文本做编码。BERT 会自动把文本处理成类似 `[CLS] text [SEP] [PAD]` 的格式，返回：
   - `input_ids`：每个 token 在词表中的编号，比如 `[CLS]=101`、`[SEP]=102`、`[PAD]=0`、`[UNK]=100`。
   - `attention_mask`：真实 token 为 1，padding 为 0，让模型忽略补齐部分。
   - `labels`：文本对应的分类标签。

3. 封装 DataLoader  
   用 `DataLoader` 封装训练集、验证集和测试集，设置 `batch_size=2`，训练时 `shuffle=True`，并通过 `collate_fn` 动态把每个 batch 转成 BERT 所需张量。

可以这样总结：普通机器学习要自己做分词、停用词、TF-IDF 等特征工程，而 BERT 主要依赖 tokenizer 把文本转成 `input_ids + attention_mask`，再交给预训练模型学习上下文语义。

项目里还可以补充一个优化点：`config.py` 里设置了 `pad_size=32`，但当前 tokenizer 主要用了 `padding=True`，面试时可以说明实际工程中会加上 `truncation=True, max_length=pad_size`，避免超长文本导致显存和推理时间不稳定。

## 2. BERT 模型做迁移学习的思路及实现细节

BERT 的迁移学习思路是：先使用大规模语料预训练好的 `bert-base-chinese` 获得中文语义表示能力，然后在自己的新闻分类数据上微调，让模型适配具体的 10 分类任务。

课堂笔记里 BERT 可以概括为：

- 结构：Transformer Encoder-only，适合做语义理解任务。
- Embedding：Token Embedding、Segment Embedding、Position Embedding 三种向量相加。
- BERT Base：12 层 Encoder、12 个 Attention Head、隐藏层维度 768。
- 预训练任务：MLM 和 NSP。MLM 是 Masked Language Model，随机遮住部分词让模型预测；NSP 是 Next Sentence Prediction，让模型判断两个句子是否连续。

我代码中的实现是：

1. 加载本地预训练模型  
   在 `Config` 里通过 `BertModel.from_pretrained(self.bert_path, local_files_only=True)` 加载本地 `bert-base-chinese`，同时加载 tokenizer 和 config。

2. 加分类头  
   在 `BERTClas` 里使用：

   ```python
   self.bert = conf.bert_model
   self.fc = nn.Linear(conf.hidden_size, conf.num_classes)
   ```

   BERT 输出的 `pooled_output` 可以理解为 `[CLS]` 位置的整句表示，维度是 768；再经过全连接层映射到 10 个类别。

3. 端到端微调  
   当前代码没有冻结 BERT 参数，而是把 BERT 主体和分类层一起交给 AdamW 更新。这种方式比只训练分类头效果更好，但训练成本也更高。

面试可以这样说：我的项目不是从零训练 BERT，而是利用预训练模型已经学到的中文语义能力，只在任务数据上微调分类层和 BERT 参数，这样数据量不大时也能获得比随机森林、FastText 更强的语义理解效果。

## 3. BERT 模型完成训练及预测的思路

训练流程可以按课堂笔记里的“225 原则”讲：

1. 两个初始化  
   初始化模型、损失函数和优化器。项目中模型是 `BERTClas()`，损失函数是 `CrossEntropyLoss`，优化器是 `AdamW(model.parameters(), lr=5e-5)`。

2. 两层遍历  
   外层遍历 epoch，内层遍历 DataLoader 的 batch。项目配置中 `num_epochs=2`，`batch_size=2`。

3. 五个训练步骤  
   每个 batch 中执行：
   - 前向传播：`logits = model(input_ids, attention_mask)`
   - 计算损失：`loss = criterion(logits, labels)`
   - 梯度清零：`optimizer.zero_grad()`
   - 反向传播：`loss.backward()`
   - 参数更新：`optimizer.step()`

验证和保存模型的流程是：

- 用 `model.eval()` 切换到评估模式。
- 用 `torch.no_grad()` 关闭梯度计算，提高速度并节省显存。
- 对每个 batch 前向传播，取 `argmax(logits, dim=1)` 得到预测类别。
- 用 `classification_report、f1_score、accuracy_score、precision_score` 评估。
- 如果验证集 `micro F1` 超过历史最优，就用 `torch.save(model.state_dict(), conf.model_save_path)` 保存模型。

预测流程是：

1. 加载训练好的模型权重，如 `bert20250521_.pt`。
2. 对用户输入文本调用 tokenizer，得到 `input_ids` 和 `attention_mask`。
3. `model.eval()` 加 `torch.no_grad()` 做推理。
4. 对 logits 做 `softmax`，再 `argmax` 得到类别 id。
5. 用 `class_list[pred_id]` 映射成具体类别。
6. Flask 部署 `/predict` 接口，客户端传入 `{"text": "..."}`，服务端返回 `{"text": "...", "pred_class": "education"}` 这样的 JSON。

课堂笔记中的项目对比结果可以作为面试补充：随机森林精度约 82.47%，FastText 约 91.65%，BERT 约 93.64%，说明 BERT 在语义理解上效果更强，但推理速度比 FastText 慢。

## 4. 对大模型及国产化算力平台的了解

大模型通常指参数规模达到十亿级以上的模型，常见单位有 M、B、T，分别表示百万、十亿、万亿。和传统 NLP 模型相比，大模型的特点是参数量大、训练数据大、泛化能力强，能通过提示词完成分类、抽取、生成、问答、代码等任务。

从结构上看，BERT 偏 Encoder，适合理解类任务；GPT、DeepSeek、Qwen、GLM、Kimi 等大语言模型多数偏生成式，常用于对话、文本生成和复杂推理。多模态大模型还可以处理图像、语音、视频和传感器数据，比如 Stable Diffusion、FLUX、Qwen2.5-VL、GLM-4V。

大模型落地常见路线有：

- 提示词工程：不训练模型，通过 prompt 约束输出。
- LoRA 微调：只训练少量低秩适配参数，降低显存和训练成本。
- RAG：把外部知识库检索结果拼进 prompt，解决知识更新和事实依据问题。
- Agent：让模型具备工具调用、任务规划和多步执行能力。

国产化算力平台方面，核心是用国产 AI 芯片和软件栈替代或补充 CUDA 生态。工程上关注的不只是“能不能跑”，还包括框架适配、算子支持、混合精度、显存利用、分布式通信、吞吐量和部署稳定性。比如在国产 NPU 环境中，要关注驱动、CANN、MindSpore 或 PyTorch NPU 适配、HCCL 通信、FP16/BF16 精度、算子兼容和模型导出部署。

面试可以这样总结：我理解国产化算力平台的难点在于软硬件协同。模型代码要适配设备，训练和推理要关注显存、batch size、并行策略、量化压缩和算子兼容；业务上则关注成本、数据安全、供应链可控和私有化部署能力。

## 5. DeepSeek 实现文本分类的思路及实现细节

DeepSeek 做文本分类可以走提示词工程路线，不一定先训练模型。核心思路是把“分类任务说明、候选标签、输出格式、待分类文本”组织成 prompt，让 DeepSeek 直接输出类别。

实现流程可以分为 6 步：

1. 准备类别集合  
   使用和项目一致的 10 个类别：`finance、realty、stocks、education、science、society、politics、sports、game、entertainment`。

2. 设计提示词模板  
   在 prompt 中明确告诉模型：你是文本分类助手，只能从给定类别中选择一个，不能输出解释，最好只输出类别名或类别 id。

   示例：

   ```text
   你是一个中文新闻文本分类模型。
   可选类别：finance, realty, stocks, education, science, society, politics, sports, game, entertainment。
   请只返回一个类别英文名，不要输出解释。
   文本：{query}
   ```

3. 使用 zero-shot 或 few-shot  
   zero-shot 是只给规则不给样例；few-shot 是在 prompt 中加入几条示例，比如“学校招生新闻 -> education”“足球比赛新闻 -> sports”，可以提升稳定性。

4. 调用 DeepSeek API  
   用低温度参数，比如 `temperature=0` 或较小值，让输出更稳定。拿到回复后做字符串清洗和标签映射。

5. 做输出校验  
   因为大模型可能输出解释或不在候选集合内的标签，所以需要白名单校验。如果输出不属于 10 个类别，就重新请求，或者让模型按 JSON 格式重答。

6. 模型评估  
   和 BERT 一样，在测试集上循环预测，统计 `accuracy、precision、recall、F1、classification_report`。

DeepSeek 方案的优点是开发快、无需训练、类别解释能力强；缺点是速度慢、成本高、输出不如 BERT 稳定，容易受 prompt 影响。课堂笔记里的对比也能说明：BERT 本地推理精度高、速度较稳定，而 LLM 提示词分类更适合快速验证、冷启动或少样本场景。

如果要进一步提升 DeepSeek 分类效果，可以用 LoRA 做指令微调，或者用 RAG 加入类别定义和业务规则；如果追求线上高并发，BERT 或 FastText 这类本地模型通常更适合。

## 6. 对大模型幻觉问题解决思路的理解

大模型幻觉是指模型生成了看起来合理但事实错误、没有依据或不符合约束的内容。在文本分类任务里，幻觉可能表现为：输出不存在的类别、编造解释、把不确定内容强行归类，或者没有按要求返回结构化结果。

产生幻觉的原因主要有：

- 大模型本质是根据上下文预测下一个 token，不等于真实理解事实。
- 训练知识有边界，遇到未见过或过时信息容易猜测。
- prompt 约束不清晰时，模型会自由发挥。
- 解码参数过高，比如 temperature 太高，会增加随机性。
- 没有外部知识或规则校验，输出缺少事实约束。

解决思路可以从 6 个层面讲：

1. Prompt 约束  
   明确候选范围和输出格式，比如“只能从 10 个类别中选一个，只输出英文标签，不要解释”。

2. 结构化输出  
   使用 JSON schema、函数调用或固定模板，让模型必须返回 `{"label": "education"}` 这样的结构。

3. 白名单校验  
   对输出类别做程序校验，不在候选类别内就重试或走兜底模型。

4. 降低随机性  
   分类任务中设置较低的 `temperature` 和 `top_p`，提高输出一致性。

5. RAG 或外部知识约束  
   对知识问答类任务，把检索到的权威资料放进上下文，并要求模型基于资料回答。

6. 人工审核和置信度机制  
   对低置信度、敏感业务或高风险结果进入人工复核；线上记录错误样本，持续更新 prompt、规则或微调数据。

结合我的项目可以这样回答：如果用 DeepSeek 做分类，我会先用明确标签列表和低温度降低幻觉，再用程序做标签白名单校验；如果模型输出非法标签，就让它重新按固定格式回答。对线上高并发场景，我会优先使用 BERT 这类本地分类模型作为稳定主链路，大模型用于解释、冷启动或辅助标注。
