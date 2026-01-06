# MeaCap Retrieve-then-Filter 模块详细解析

## 一、模块概述

### 1.1 什么是 Retrieve-then-Filter？

**Retrieve-then-Filter** 是 MeaCap 论文提出的核心模块，用于从大规模记忆库中提取与图像相关的关键概念。

**设计理念**：
- **Retrieve（检索）**：从记忆库中找到与当前图像最相似的描述
- **Filter（过滤）**：从这些描述中提取关键概念（实体、属性、关系）

### 1.2 为什么需要这个模块？

**ViECap 原始方法的局限性**：
- 使用预定义的实体词表（如 COCO 的 80 个类别）
- 词表固定，无法适应新领域或细粒度概念
- 只能检测单个词，无法捕获短语级别的概念（如 "cute girl"）

**MeaCap 的优势**：
- 从大规模记忆库（如 COCO 训练集）中动态检索
- 提取细粒度的短语级概念（如 "cute girl"、"wooden table"）
- 通过场景图解析和语义合并，提高概念质量

---

## 二、模块架构

### 2.1 整体流程

```
输入图像
    ↓
[Retrieve 阶段]
    ├─ CLIP 编码图像
    ├─ 计算与记忆库的相似度
    └─ 检索 Top-K 记忆描述
    ↓
[Filter 阶段]
    ├─ 场景图解析（Flan-T5）
    ├─ 实体提取与统计
    ├─ 语义合并（SentenceBERT）
    └─ 图像相关性过滤
    ↓
输出关键概念（List[str]）
```

### 2.2 两个阶段的详细说明

---

## 三、Retrieve 阶段详解

### 3.1 目标

从大规模记忆库中找到与当前图像最相似的 K 条文本描述。

### 3.2 实现步骤

#### 步骤 1：准备记忆库

**文件结构**：
```
data/memory/{memory_id}/
├── memory_captions.json          # 文本描述列表
├── memory_clip_embeddings.pt     # CLIP 嵌入向量 (N, 512)
└── memory_wte_embeddings.pt      # SentenceBERT 嵌入向量 (N, 384)
```

**代码位置**：`viecap_inference_adapted.py:87-95`

```python
memory_caption_path = os.path.join(f"data/memory/{memory_id}", "memory_captions.json")
memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_clip_embeddings.pt")
memory_wte_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_wte_embeddings.pt")

memory_clip_embeddings = torch.load(memory_clip_embedding_file)  # (N, 512)
memory_wte_embeddings = torch.load(memory_wte_embedding_file)    # (N, 384)
with open(memory_caption_path, 'r') as f:
    memory_captions = json.load(f)  # List[str], length = N
```

**说明**：
- `N`：记忆库大小（如 COCO 训练集约 118K 条描述）
- CLIP 嵌入：用于图像-文本相似度计算
- SentenceBERT 嵌入：用于后续的语义合并

#### 步骤 2：编码当前图像

**代码位置**：`viecap_inference_adapted.py:112`

```python
batch_image_embeds = vl_model.compute_image_representation_from_image_path(args.image_path)
# 输出: (1, 512) - CLIP 图像嵌入
```

**说明**：
- 使用 CLIP 的视觉编码器提取图像特征
- 与记忆库的 CLIP 嵌入在同一空间

#### 步骤 3：计算相似度

**代码位置**：`viecap_inference_adapted.py:115-123`

```python
clip_score, _ = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
    batch_image_embeds,      # (1, 512)
    memory_clip_embeddings    # (N, 512)
)
# 输出: clip_score (1, N) - 每个记忆描述的相似度分数
```

**计算方式**：
- 余弦相似度：`cosine_similarity(image_embed, memory_embed)`
- 分数范围：[-1, 1]，越高越相似

#### 步骤 4：Top-K 检索

**代码位置**：`viecap_inference_adapted.py:124-125`

```python
select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
# 输出: (K,) - Top-K 记忆描述的索引

select_memory_captions = [memory_captions[id] for id in select_memory_ids]
# 输出: List[str], length = K - Top-K 记忆描述文本
```

**示例**（K=5）：
```python
select_memory_captions = [
    "A cute girl is sitting on a bed with a pink blanket.",
    "A young woman lies on a bed covered with a pink blanket.",
    "A girl is resting on a bed with pink sheets.",
    "A person is sitting on a bed with a pink blanket.",
    "A young girl is on a bed with a pink blanket."
]
```

### 3.3 Retrieve 阶段总结

**输入**：
- 当前图像
- 记忆库（文本 + CLIP 嵌入）

**输出**：
- Top-K 记忆描述（文本）

**关键点**：
- 使用 CLIP 的跨模态能力进行检索
- 检索的是**完整的句子**，而不是单个词

---

## 四、Filter 阶段详解

### 4.1 目标

从 Top-K 记忆描述中提取关键概念（实体、属性、关系）。

### 4.2 实现步骤

#### 步骤 1：场景图解析（Flan-T5）

**代码位置**：`utils/detect_utils.py` → `retrieve_concepts` → `parse`

**功能**：将自然语言描述转换为 Textual Scene Graph

**输入**：
```python
select_memory_captions = [
    "A cute girl is sitting on a bed with a pink blanket.",
    "A young woman lies on a bed covered with a pink blanket.",
    ...
]
```

**输出**（场景图格式）：
```
Scene Graph 1:
  Objects: [girl, bed, blanket]
  Attributes: [cute, pink]
  Relations: [sitting on, with]

Scene Graph 2:
  Objects: [woman, bed, blanket]
  Attributes: [young, pink]
  Relations: [lies on, covered with]
```

**实现**：
- 使用 Flan-T5 模型（`lizhuang144/flan-t5-base-VG-factual-sg`）
- 专门训练用于场景图解析
- 输入：自然语言描述
- 输出：结构化的场景图文本

#### 步骤 2：实体提取与统计

**代码位置**：`utils/detect_utils.py` → `get_graph_dict`

**功能**：
1. 从场景图中提取实体（对象、属性、关系）
2. 统计每个实体在不同描述中的出现频率
3. 计算每个实体的 SentenceBERT 嵌入

**输出**：
```python
entities_ = [
    "cute girl", "girl", "young woman", "woman",
    "bed", "pink blanket", "blanket",
    ...
]
count_dict_ = {
    "girl": 3,
    "bed": 5,
    "blanket": 4,
    "pink": 3,
    ...
}
entire_graph_dict = {
    "girl": [sentence_bert_embed_1, sentence_bert_embed_2, ...],
    "bed": [sentence_bert_embed_1, ...],
    ...
}
```

#### 步骤 3：语义合并（SentenceBERT）

**代码位置**：`utils/detect_utils.py` → `merge_graph_dict_new`

**功能**：合并语义相似的实体

**示例**：
```python
# 合并前
entities = ["girl", "cute girl", "young girl", "woman", "young woman"]

# 合并后（相似度阈值：0.8）
concepts = ["girl", "woman", "bed", "blanket"]
```

**合并规则**：
1. 计算实体之间的 SentenceBERT 相似度
2. 如果相似度 > 阈值（如 0.8），则合并
3. 保留出现频率最高的实体作为代表

**优势**：
- 减少冗余（"girl"、"cute girl"、"young girl" → "girl"）
- 提高概念质量

#### 步骤 4：图像相关性过滤

**代码位置**：`utils/detect_utils.py` → `retrieve_concepts`

**功能**：根据图像特征过滤不相关的实体

**方法**：
1. 计算每个实体的 SentenceBERT 嵌入
2. 计算实体与图像的 CLIP 相似度（通过文本-图像对齐）
3. 过滤低相似度的实体

**输出**：
```python
detected_objects = ["cute girl", "bed"]  # Top-4 概念
```

### 4.3 Filter 阶段总结

**输入**：
- Top-K 记忆描述（文本）

**输出**：
- 关键概念列表（`List[str]`，通常 3-4 个）

**关键点**：
- 场景图解析：从句子中提取结构化信息
- 语义合并：减少冗余，提高质量
- 图像过滤：确保概念与图像相关

---

## 五、如何替换到 ViECap

### 5.1 替换位置

**原始代码**（`validation.py:136-138`）：
```python
# ViECap 原始实体检测
logits = image_text_simiarlity(
    texts_embeddings,           # 预定义实体词表的 CLIP 嵌入
    temperature=args.temperature,
    images_features=image_features
)
detected_objects, _ = top_k_categories(
    entities_text,              # 预定义实体词表（如 COCO 80 类）
    logits,
    args.top_k,                  # Top-K 类别
    args.threshold               # 相似度阈值
)
detected_objects = detected_objects[0]  # List[str]
```

**MeaCap 替换**（`validation_meacap.py:67-88`）：
```python
# MeaCap Retrieve-then-Filter
# Retrieve 阶段
batch_image_embeds = vl_model.compute_image_representation_from_image_path(image_path)
clip_score, _ = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
    batch_image_embeds, memory_clip_embeddings
)
select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
select_memory_captions = [memory_captions[id] for id in select_memory_ids]

# Filter 阶段
detected_objects = retrieve_concepts(
    parser_model=parser_model,
    parser_tokenizer=parser_tokenizer,
    wte_model=wte_model,
    select_memory_captions=select_memory_captions,
    image_embeds=batch_image_embeds,
    device=device
)  # List[str]
```

### 5.2 接口兼容性

**关键点**：两个版本的输出格式完全一致

```python
# 原始版本输出
detected_objects = ["person", "bed"]  # List[str]

# MeaCap 版本输出
detected_objects = ["cute girl", "bed"]  # List[str]
```

**后续处理完全相同**：
```python
# 两个版本都使用相同的代码
discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects)
discrete_embeddings = model.word_embed(discrete_tokens)
# ... 提示组合、文本生成等
```

### 5.3 为什么是 Plug-and-Play？

**条件 1：输入兼容**
- 原始版本：需要图像特征 + 预定义实体词表
- MeaCap 版本：需要图像特征 + 记忆库
- ✅ 两者都只需要图像特征作为输入

**条件 2：输出兼容**
- 原始版本：`List[str]`（实体列表）
- MeaCap 版本：`List[str]`（概念列表）
- ✅ 输出格式完全一致

**条件 3：接口不变**
- 后续代码（`compose_discrete_prompts`、`word_embed`、`beam_search` 等）完全不变
- ✅ 只需替换实体检测这一小块代码

---

## 六、代码实现细节

### 6.1 核心函数调用链

```
main()
  ↓
validation_coco_flickr30k_meacap()
  ↓
[Retrieve]
  vl_model.compute_image_representation_from_image_path()
  vl_model_retrieve.compute_image_text_similarity_via_embeddings()
  clip_score.topk()
  ↓
[Filter]
  retrieve_concepts()
    ├─ parse()                    # 场景图解析
    ├─ get_graph_dict()           # 实体提取
    ├─ merge_graph_dict_new()     # 语义合并
    └─ 图像相关性过滤
  ↓
detected_objects (List[str])
```

### 6.2 关键文件

| 文件 | 功能 |
|------|------|
| `utils/detect_utils.py` | `retrieve_concepts` 主函数 |
| `utils/parse_tool.py` | 场景图解析工具 |
| `models/clip_utils.py` | CLIP 工具类（检索用） |
| `validation_meacap.py` | 批量评估脚本 |

### 6.3 内存优化

**大型记忆库处理**（CC3M/SS1M）：
```python
if memory_id == 'cc3m' or memory_id == 'ss1m':
    retrieve_on_CPU = True
    vl_model_retrieve = copy.deepcopy(vl_model).to(cpu_device)
    memory_clip_embeddings = memory_clip_embeddings.to(cpu_device)
```

**说明**：
- 大型记忆库（数百万条）无法全部加载到 GPU
- 检索阶段在 CPU 上进行
- 后续处理仍在 GPU 上进行

---

## 七、性能分析

### 7.1 计算复杂度

**Retrieve 阶段**：
- 图像编码：O(1)
- 相似度计算：O(N)，N = 记忆库大小
- Top-K 检索：O(N log K)

**Filter 阶段**：
- 场景图解析：O(K × L)，K = 检索数量，L = 描述长度
- 实体提取：O(K × E)，E = 每个描述的平均实体数
- 语义合并：O(E²)
- 图像过滤：O(E)

**总复杂度**：O(N + K × L + E²)

### 7.2 实际性能

**COCO 记忆库**（N ≈ 118K）：
- Retrieve：~100ms（GPU）
- Filter：~200-500ms（取决于 K 和描述长度）
- 总计：~300-600ms per image

**优化建议**：
- 使用预提取的图像特征（`--using_image_features`）
- 减少检索数量（`--memory_caption_num`）
- 批量处理（在 `validation_meacap.py` 中已实现）

---

## 八、优势与局限性

### 8.1 优势

✅ **细粒度概念**：
- 提取短语级概念（"cute girl"）而非单词（"girl"）
- 保留语义信息（"pink blanket" vs "blanket"）

✅ **动态适应**：
- 不依赖预定义词表
- 可以从任何记忆库中提取概念

✅ **语义理解**：
- 通过场景图解析理解句子结构
- 通过语义合并减少冗余

### 8.2 局限性

⚠️ **计算开销**：
- 需要场景图解析（Flan-T5）
- 需要语义相似度计算（SentenceBERT）
- 比原始方法慢

⚠️ **概念格式**：
- 提取的是短语，可能包含修饰词
- 直接用于硬提示可能导致语法问题（如 "cute girl" 缺少冠词）

⚠️ **依赖记忆库**：
- 需要准备记忆库文件
- 记忆库质量影响概念提取质量

---

## 九、使用建议

### 9.1 参数调优

**`memory_caption_num`**（检索数量）：
- 较小值（3-5）：速度快，但可能遗漏重要概念
- 较大值（10-20）：更全面，但计算开销大
- **推荐**：5-10

**`soft_prompt_first`**（提示顺序）：
- `False`：硬提示在前（默认）
- `True`：软提示在前（可能改善语法）
- **推荐**：根据实验结果选择

### 9.2 记忆库选择

**COCO 记忆库**：
- 适合 COCO 数据集评估
- 大小：~118K 条描述
- 质量：高（人工标注）

**其他记忆库**：
- CC3M/SS1M：更大规模，但质量可能较低
- 自定义记忆库：可以针对特定领域

### 9.3 性能优化

1. **使用预提取特征**：`--using_image_features`
2. **减少检索数量**：`--memory_caption_num 3`
3. **批量处理**：在 `validation_meacap.py` 中已实现
4. **GPU 优化**：确保 CLIP、Flan-T5、SentenceBERT 在 GPU 上

---

## 十、总结

### 10.1 核心思想

**Retrieve-then-Filter** 通过两个阶段从记忆库中提取关键概念：
1. **Retrieve**：找到与图像最相似的描述
2. **Filter**：从描述中提取关键概念

### 10.2 替换方式

**Plug-and-Play**：
- 只需替换实体检测部分
- 输出格式完全兼容
- 其他代码无需修改

### 10.3 实际应用

- ✅ 单图推理：`viecap_inference_adapted.py`
- ✅ 批量评估：`validation_meacap.py`
- ✅ 性能对比：与原始版本对比评估指标

### 10.4 下一步

1. 运行评估并对比性能
2. 分析概念提取质量
3. 根据结果调整参数
4. 考虑后处理优化（如概念规范化）

