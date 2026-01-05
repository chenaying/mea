# MeaCap 论文源码解析

## 一、项目概述

**MeaCap** (Memory-Augmented Zero-shot Image Captioning) 是 CVPR 2024 的论文，提出了一种基于记忆增强的零样本图像描述生成框架。

### 核心思想

- **问题**：零样本图像描述生成存在两个问题
  - 训练无关方法（Training-free）：容易产生幻觉（hallucinations）
  - 仅文本训练（Text-only-training）：容易失去泛化能力

- **解决方案**：引入文本记忆库（Memory Bank），通过检索-过滤机制获取与图像高度相关的关键概念

### 论文信息

- **会议**：CVPR 2024
- **arXiv**：2403.03715
- **GitHub**：https://github.com/joeyz0z/MeaCap

## 二、代码架构

### 2.1 项目结构

```
MeaCap-main/
├── models/              # 核心模型实现
│   ├── main.py          # 主训练/推理脚本
│   ├── bart.py          # BART模型封装
│   ├── clip_utils.py    # CLIP工具类
│   └── ...
├── utils/               # 工具函数
│   ├── detect_utils_new.py    # 关键词检测（检索-过滤）
│   ├── parse_tool_new.py      # 场景图解析
│   ├── generate_utils.py      # 文本生成工具
│   └── ...
├── dataset/             # 数据集处理
│   ├── ImgDataset.py    # 图像数据集
│   └── ...
├── language_models/     # 语言模型
│   └── language_model.py
├── inference.py         # 推理脚本
├── prepare_embedding.py # 记忆库预处理
├── args.py             # 参数配置
└── README.md
```

### 2.2 核心组件

#### 1. **记忆库（Memory Bank）**

**位置**：`data/memory/`

**结构**：
```
data/memory/
├── coco/
│   ├── memory_captions.json          # 文本描述
│   ├── memory_clip_embeddings.pt     # CLIP嵌入
│   └── memory_wte_embeddings.pt      # SentenceBERT嵌入
├── cc3m/
└── ...
```

**预处理脚本**：`prepare_embedding.py`

```python
# 主要功能：
# 1. 加载文本语料库
# 2. 使用CLIP编码文本 → memory_clip_embeddings.pt
# 3. 使用SentenceBERT编码 → memory_wte_embeddings.pt
# 4. 保存为快速检索格式
```

#### 2. **检索-过滤模块（Retrieve-then-Filter）**

**核心文件**：`utils/detect_utils_new.py`

**流程**：

```python
def detect_keyword(parser_model, parser_tokenizer, wte_model, 
                  image_embeds, vl_model, select_memory_captions, ...):
    """
    从记忆库中检索并过滤关键概念
    """
    # 1. 解析记忆描述为场景图
    scene_graphs = parse(parser_model, parser_tokenizer, 
                         text_input=select_memory_captions)
    
    # 2. 提取实体和关系
    entities_, count_dict_, entire_graph_dict = get_graph_dict(
        wte_model, scene_graphs, ...)
    
    # 3. 合并相似实体，过滤与图像相关的概念
    concepts, count_dict, filtered_graph_dict = merge_graph_dict_new(
        wte_model, vl_model, image_embeds, 
        entities_, count_dict_, entire_graph_dict, ...)
    
    return concepts[:4]  # 返回Top-4概念
```

**关键步骤**：

1. **检索（Retrieve）**：
   - 使用CLIP计算图像与记忆库描述的相似度
   - 选择Top-K最相似的描述

2. **解析（Parse）**：
   - 使用SceneGraphParser将描述解析为场景图
   - 提取实体、属性、关系

3. **过滤（Filter）**：
   - 使用CLIP计算概念与图像的相似度
   - 合并相似实体
   - 选择与图像最相关的概念

#### 3. **语言模型（CBART）**

**位置**：`models/bart.py`, `src/transformers/`

**特点**：
- 基于BART的关键词到句子生成模型
- 支持文本填充（Text Infilling）
- 可训练或使用预训练模型

**两种模式**：

1. **Training-free (MeaCap_TF)**：
   - 使用预训练的CBART（One-billion-word）
   - 需要提示："The image depicts that"

2. **Text-only-training (MeaCap_ToT)**：
   - 使用在描述数据上微调的CBART
   - 无需提示

#### 4. **视觉-语言融合评分**

**位置**：`utils/generate_utils.py`

**功能**：
- 计算生成文本与图像的匹配度
- 使用CLIP和SentenceBERT进行多模态评分
- 指导文本生成过程

## 三、核心算法流程

### 3.1 整体流程

```
输入图像
    ↓
CLIP编码 → 图像特征
    ↓
检索记忆库 → Top-K相似描述
    ↓
场景图解析 → 实体、属性、关系
    ↓
CLIP过滤 → 与图像相关的概念
    ↓
关键词提取 → Top-4概念
    ↓
CBART生成 → 关键词 → 完整描述
    ↓
视觉-语言融合评分 → 优化描述
    ↓
输出描述
```

### 3.2 详细步骤

#### Step 1: 记忆库检索

```python
# models/main.py 或 inference.py
# 1. 加载记忆库
memory_clip_embeddings = torch.load('memory_clip_embeddings.pt')
memory_captions = json.load('memory_captions.json')

# 2. 计算相似度
image_embedding = vl_model.compute_image_representation(...)
clip_score = cosine_similarity(image_embedding, memory_clip_embeddings)

# 3. 选择Top-K
select_memory_ids = clip_score.topk(k=5, dim=-1)[1]
select_memory_captions = [memory_captions[id] for id in select_memory_ids]
```

#### Step 2: 场景图解析

```python
# utils/parse_tool_new.py
def parse(parser_model, parser_tokenizer, text_input, device):
    """
    使用SceneGraphParser将文本解析为场景图
    """
    # 输入：["A dog is running in the park", ...]
    # 输出：场景图结构（实体、属性、关系）
    scene_graphs = parser_model.generate(...)
    return scene_graphs
```

**场景图结构**：
```python
{
    "entities": ["dog", "park"],
    "attributes": {"dog": ["running"]},
    "relations": [("dog", "in", "park")]
}
```

#### Step 3: 概念过滤

```python
# utils/parse_tool_new.py
def merge_graph_dict_new(wte_model, vl_model, image_embeds, 
                         entities_, count_dict_, entire_graph_dict, ...):
    """
    合并相似实体，过滤与图像相关的概念
    """
    # 1. 使用SentenceBERT合并语义相似的实体
    # 2. 使用CLIP计算概念与图像的相似度
    # 3. 选择Top-K最相关的概念
    concepts = filter_by_clip_similarity(...)
    return concepts
```

#### Step 4: 文本生成

```python
# utils/generate_utils.py
def generate_function(model, tokenizer, vl_model, ...):
    """
    使用CBART从关键词生成完整描述
    """
    # 1. 将关键词编码为输入
    encoder_inputs = tokenizer(keywords, ...)
    
    # 2. CBART生成
    outputs = model.generate(
        encoder_inputs,
        max_length=max_len,
        temperature=temperature,
        ...
    )
    
    # 3. 视觉-语言融合评分优化
    scores = compute_clip_score(outputs, image_embeds)
    
    return best_caption
```

## 四、关键代码解析

### 4.1 记忆库预处理

**文件**：`prepare_embedding.py`

```python
# 主要流程
def main():
    # 1. 加载CLIP和SentenceBERT
    vl_model = CLIP(args.clip_model)
    wte_model = SentenceTransformer(args.wte_model_path)
    
    # 2. 批量处理文本
    for text_batch in textual_data:
        # CLIP编码
        clip_embeds = vl_model.compute_text_representation(text_batch)
        # SentenceBERT编码
        wte_embeds = wte_model.encode(text_batch)
    
    # 3. 保存嵌入
    torch.save(clip_embeds, 'memory_clip_embeddings.pt')
    torch.save(wte_embeds, 'memory_wte_embeddings.pt')
```

### 4.2 关键词检测

**文件**：`utils/detect_utils_new.py`

```python
def detect_keyword(parser_model, parser_tokenizer, wte_model, 
                  image_embeds, vl_model, select_memory_captions, ...):
    """
    核心函数：从记忆库中提取关键词
    """
    # 1. 解析场景图
    scene_graphs = parse(parser_model, parser_tokenizer,
                         text_input=select_memory_captions,
                         device=device)
    
    # 2. 提取实体和关系
    entities_, count_dict_, entire_graph_dict = get_graph_dict(
        wte_model, scene_graphs, type_dict, attribute_dict)
    
    # 3. 合并和过滤
    concepts, count_dict, filtered_graph_dict = merge_graph_dict_new(
        wte_model, vl_model, image_embeds, 
        entities_, count_dict_, entire_graph_dict, ...)
    
    return concepts[:4]  # 返回Top-4
```

### 4.3 文本生成

**文件**：`utils/generate_utils.py`

```python
def generate_function(model, tokenizer, vl_model, wte_model, ...):
    """
    从关键词生成完整描述
    """
    # 1. 编码关键词
    encoder_inputs = tokenizer(keywords, ...)
    
    # 2. 生成文本（支持多种策略）
    if do_sample:
        # 采样生成
        outputs = model.sample(...)
    else:
        # 贪心生成
        outputs = model.greedy_decode(...)
    
    # 3. 视觉-语言融合评分
    clip_scores = compute_clip_score(outputs, image_embeds)
    
    # 4. 选择最佳描述
    best_caption = select_best_by_score(outputs, clip_scores)
    
    return best_caption
```

### 4.4 CLIP工具类

**文件**：`models/clip_utils.py`

```python
class CLIP(nn.Module):
    def __init__(self, model_name):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def compute_image_representation(self, image_path):
        """图像编码"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        image_embeds = self.model.vision_model(...)
        return image_embeds
    
    def compute_text_representation(self, text_list):
        """文本编码"""
        inputs = self.processor(text=text_list, return_tensors="pt")
        text_embeds = self.model.text_model(...)
        return text_embeds
    
    def compute_image_text_similarity(self, image_embeds, text_embeds):
        """计算相似度"""
        similarity = cosine_similarity(image_embeds, text_embeds)
        return similarity
```

## 五、三种运行模式

### 5.1 Training-free (MeaCap_TF)

**特点**：
- 无需训练，使用预训练CBART
- 需要提示："The image depicts that"
- 支持提示集成（prompt ensembling）

**运行命令**：
```bash
python inference.py \
    --use_prompt \
    --memory_id cc3m \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_one_billion
```

**代码位置**：`inference.py`

### 5.2 Text-only-training (MeaCap_ToT)

**特点**：
- 使用在描述数据上微调的CBART
- 无需提示
- 性能更好

**运行命令**：
```bash
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO
```

### 5.3 Memory + ViECAP (MeaCap_InvLM)

**特点**：
- 将记忆概念集成到ViECAP
- 即插即用方式
- 替换ViECAP的实体模块

**运行命令**：
```bash
python viecap_inference.py \
    --memory_id coco \
    --image_path "*.jpg" \
    --weight_path "checkpoints/train_coco/coco_prefix-0014.pt"
```

**代码位置**：`viecap_inference.py`

## 六、数据流程

### 6.1 训练数据准备

1. **准备文本语料库**：
   - COCO、Flickr30k、CC3M、SS1M等
   - 格式：JSON列表 `["caption1", "caption2", ...]`

2. **预处理记忆库**：
```bash
python prepare_embedding.py \
    --memory_id coco \
    --memory_path data/memory/coco/memory_captions.json
```

3. **生成嵌入文件**：
   - `memory_clip_embeddings.pt`：CLIP文本嵌入
   - `memory_wte_embeddings.pt`：SentenceBERT嵌入

### 6.2 推理流程

```
图像输入
    ↓
CLIP编码图像
    ↓
检索记忆库（Top-5描述）
    ↓
场景图解析（实体、属性、关系）
    ↓
CLIP过滤（Top-4概念）
    ↓
CBART生成（关键词→句子）
    ↓
视觉-语言融合评分
    ↓
输出描述
```

## 七、关键参数配置

### 7.1 记忆库参数

```python
# args.py
--memory_id: str = "coco"           # 记忆库ID
--memory_caption_path: str          # 记忆库路径
--memory_caption_num: int = 5      # 检索的描述数量
```

### 7.2 生成参数

```python
--max_len: int = 20                 # 最大生成长度
--min_len: int = 10                 # 最小生成长度
--temperature: float = 1.0          # 温度参数
--top_k: int = 0                    # Top-K采样
--top_p: float = 0.9                # 核采样
--repetition_penalty: float = 2.0    # 重复惩罚
```

### 7.3 融合评分参数

```python
--alpha: float = 0.1                # 流畅度权重
--beta: float = 0.8                 # 图像匹配度权重
--gamma: float = 0.2                # 其他权重
```

## 八、模型性能

### 8.1 零样本描述生成

| 方法 | 训练 | 记忆库 | MSCOCO CIDEr | NoCaps (Overall) |
|------|------|--------|--------------|-----------------|
| ConZIC | ✗ | ✗ | 5.0 | 17.5 |
| CLIPRe | ✗ | CC3M | 25.6 | 28.2 |
| **MeaCap_TF** | ✗ | CC3M | **42.5** | **40.2** |
| **MeaCap_ToT** | CC3M | CC3M | **48.3** | **45.1** |
| **MeaCap_ToT** | SS1M | SS1M | **54.9** | **47.3** |

### 8.2 跨域描述生成

| 任务 | MeaCap_TF | MeaCap_ToT | MeaCap_InvLM |
|------|-----------|------------|--------------|
| COCO | 56.9 | 84.8 | **95.4** |
| Flickr30k | 36.5 | 50.2 | **59.4** |
| COCO→Flickr30k | 34.4 | 40.3 | **43.9** |
| Flickr30k→COCO | 46.4 | 51.7 | **56.4** |

## 九、关键创新点

### 9.1 检索-过滤机制

- **检索**：使用CLIP快速检索与图像相似的描述
- **过滤**：使用场景图解析和CLIP过滤，提取关键概念
- **优势**：既保证相关性，又减少幻觉

### 9.2 记忆增强

- **文本记忆库**：大规模文本语料库的嵌入表示
- **快速检索**：预计算嵌入，支持快速相似度计算
- **灵活扩展**：可以轻松添加新的记忆库

### 9.3 视觉-语言融合评分

- **多模态评分**：结合CLIP和SentenceBERT
- **生成指导**：实时优化生成过程
- **减少幻觉**：确保生成内容与图像一致

## 十、与ViECap的对比

### 10.1 相同点

- 都使用CLIP进行图像-文本对齐
- 都提取关键概念指导生成
- 都支持零样本跨域迁移

### 10.2 不同点

| 特性 | ViECap | MeaCap |
|------|--------|--------|
| 概念来源 | 预定义实体词汇表 | 记忆库检索 |
| 提取方式 | CLIP相似度检索 | 检索-过滤机制 |
| 语言模型 | GPT-2 | CBART |
| 训练方式 | 端到端训练 | 训练无关/仅文本训练 |
| 记忆机制 | 无 | 文本记忆库 |

### 10.3 集成方式

MeaCap可以集成到ViECap：
- 替换ViECap的实体检测模块
- 使用MeaCap的检索-过滤机制
- 保持ViECap的其他组件不变

## 十一、运行示例

### 11.1 准备环境

```bash
# 安装依赖
pip install -r requirements.txt

# 下载预训练模型
# - CBART (One-billion-word / COCO / ...)
# - CLIP
# - SceneGraphParser
# - SentenceBERT
```

### 11.2 准备记忆库

```bash
# 下载预处理的记忆库（推荐）
# 或自己预处理
python prepare_embedding.py \
    --memory_id coco \
    --memory_path data/memory/coco/memory_captions.json
```

### 11.3 运行推理

```bash
# Training-free模式
python inference.py \
    --use_prompt \
    --memory_id cc3m \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_one_billion

# Text-only-training模式
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path ./checkpoints/CBART_COCO
```

## 十二、代码关键点总结

### 12.1 核心文件

1. **inference.py**：主推理脚本
2. **utils/detect_utils_new.py**：关键词检测（检索-过滤）
3. **utils/parse_tool_new.py**：场景图解析
4. **utils/generate_utils.py**：文本生成
5. **models/clip_utils.py**：CLIP工具类
6. **prepare_embedding.py**：记忆库预处理

### 12.2 关键函数

- `detect_keyword()`：从记忆库提取关键词
- `parse()`：场景图解析
- `merge_graph_dict_new()`：合并和过滤概念
- `generate_function()`：文本生成
- `compute_image_text_similarity()`：相似度计算

### 12.3 数据流

```
图像 → CLIP编码 → 检索记忆库 → 场景图解析 → 
概念过滤 → 关键词提取 → CBART生成 → 融合评分 → 输出
```

## 十三、参考文献

```bibtex
@article{zeng2024meacap,
  title={MeaCap: Memory-Augmented Zero-shot Image Captioning},
  author={Zeng, Zequn and Xie, Yan and Zhang, Hao and Chen, Chiyu and Wang, Zhengjue and Chen, Bo},
  journal={arXiv preprint arXiv:2403.03715},
  year={2024}
}
```

---

**总结**：MeaCap通过引入文本记忆库和检索-过滤机制，有效解决了零样本图像描述生成中的幻觉问题和泛化能力问题，在多个数据集上取得了优异的性能。


