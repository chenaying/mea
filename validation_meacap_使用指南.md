# validation_meacap.py 使用指南

## 一、概述

`validation_meacap.py` 是使用 **MeaCap Retrieve-then-Filter 模块**替换 ViECap 原始实体检测的批量评估脚本。

### 1.1 主要特点

- ✅ **Plug-and-Play 替换**：将 ViECap 的 `top_k_categories` 替换为 MeaCap 的 `retrieve_concepts`
- ✅ **批量评估**：支持 COCO、Flickr30k 数据集的批量评估
- ✅ **结果区分**：输出文件带有 `_meacap` 后缀，便于与原始版本对比
- ✅ **兼容性**：保持与原始 `validation.py` 相同的接口和参数

### 1.2 与原版的区别

| 特性 | validation.py (原版) | validation_meacap.py (MeaCap版) |
|------|---------------------|-------------------------------|
| **实体检测** | `top_k_categories` (CLIP + 预定义词表) | `retrieve_concepts` (Retrieve-then-Filter) |
| **输入** | 预定义实体词表 | 记忆库 (memory bank) |
| **输出文件** | `coco_generated_captions.json` | `coco_generated_captions_meacap.json` |
| **依赖模块** | `retrieval_categories.py` | `utils/detect_utils.py`, `models/clip_utils.py` |

---

## 二、前置条件

### 2.1 必需文件

1. **记忆库文件**（在 `data/memory/{memory_id}/` 目录下）：
   - `memory_captions.json`：记忆库文本描述
   - `memory_clip_embeddings.pt`：CLIP 嵌入向量
   - `memory_wte_embeddings.pt`：SentenceBERT 嵌入向量

2. **MeaCap 模块文件**：
   - `utils/detect_utils.py`：`retrieve_concepts` 函数
   - `models/clip_utils.py`：CLIP 工具类
   - `utils/parse_tool.py`：场景图解析工具

3. **模型文件**（可选，支持本地路径）：
   - Flan-T5 解析器：`checkpoints/flan-t5-base-VG-factual-sg/`
   - SentenceBERT：`checkpoints/all-MiniLM-L6-v2/`
   - CLIP（检索用）：`checkpoints/clip-vit-base-patch32/`

### 2.2 依赖包

```bash
pip install sentence-transformers
pip install transformers
pip install nltk
# 下载 NLTK 数据
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

---

## 三、使用方法

### 3.1 基本用法

```bash
python validation_meacap.py \
    --device cuda:0 \
    --clip_model ViT-B/32 \
    --language_model gpt2 \
    --name_of_datasets coco \
    --path_of_val_datasets ./annotations/coco/test_captions.json \
    --image_folder ./annotations/coco/val2014/ \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt \
    --out_path ./checkpoints/train_coco \
    --using_hard_prompt \
    --soft_prompt_first \
    --memory_id coco \
    --memory_caption_path data/memory/coco/memory_captions.json \
    --memory_caption_num 5
```

### 3.2 使用评估脚本（推荐）

```bash
# 使用 eval_coco_meacap.sh
bash scripts/eval_coco_meacap.sh train_coco 0 '' 14

# 参数说明：
# train_coco: 实验名称
# 0: GPU 设备号
# '': 其他参数（可选）
# 14: epoch 编号
```

### 3.3 参数说明

#### ViECap 原始参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `cuda:0` | GPU 设备 |
| `--clip_model` | `ViT-B/32` | CLIP 模型（用于图像编码） |
| `--language_model` | `gpt2` | 语言模型 |
| `--name_of_datasets` | `coco` | 数据集名称 |
| `--path_of_val_datasets` | `./annotations/coco/test_captions.json` | 验证集标注文件 |
| `--image_folder` | `./annotations/coco/val2014/` | 图像文件夹 |
| `--weight_path` | `./checkpoints/train_coco/coco_prefix-0014.pt` | 模型权重路径 |
| `--out_path` | `./checkpoints/train_coco` | 输出目录 |
| `--using_hard_prompt` | `False` | 使用硬提示 |
| `--soft_prompt_first` | `False` | 软提示在前 |
| `--beam_width` | `5` | Beam Search 宽度 |

#### MeaCap 特定参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vl_model` | `openai/clip-vit-base-patch32` | CLIP 模型（用于检索） |
| `--parser_checkpoint` | `lizhuang144/flan-t5-base-VG-factual-sg` | Flan-T5 解析器路径 |
| `--wte_model_path` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceBERT 模型路径 |
| `--memory_id` | `coco` | 记忆库 ID |
| `--memory_caption_path` | `data/memory/coco/memory_captions.json` | 记忆库文本文件 |
| `--memory_caption_num` | `5` | 检索的记忆描述数量 |
| `--offline_mode` | `False` | 离线模式（使用本地模型） |

---

## 四、输出结果

### 4.1 输出文件

**位置**：`{out_path}/{name_of_datasets}_generated_captions_meacap.json`

**示例**：
- `checkpoints/train_coco/coco_generated_captions_meacap.json`

### 4.2 文件格式

```json
[
    {
        "split": "valid",
        "image_name": "COCO_val2014_000000042.jpg",
        "captions": [
            "A woman throwing a frisbee in a park.",
            "A girl is throwing a frisbee in a grassy field.",
            ...
        ],
        "prediction": "A woman is throwing a frisbee in a park."
    },
    ...
]
```

### 4.3 评估指标

使用 `cocoeval.py` 计算评估指标：

```bash
python evaluation/cocoeval.py --result_file_path checkpoints/train_coco/coco*_meacap.json
```

**输出指标**：
- CIDEr
- BLEU@1, BLEU@2, BLEU@3, BLEU@4
- METEOR
- SPICE
- ROUGE-L

---

## 五、对比原始版本

### 5.1 运行原始版本

```bash
# 使用原始 validation.py
bash scripts/eval_coco.sh train_coco 0 '' 14

# 输出文件：checkpoints/train_coco/coco_generated_captions.json
```

### 5.2 运行 MeaCap 版本

```bash
# 使用 validation_meacap.py
bash scripts/eval_coco_meacap.sh train_coco 0 '' 14

# 输出文件：checkpoints/train_coco/coco_generated_captions_meacap.json
```

### 5.3 对比结果

```bash
# 评估原始版本
python evaluation/cocoeval.py --result_file_path checkpoints/train_coco/coco_generated_captions.json

# 评估 MeaCap 版本
python evaluation/cocoeval.py --result_file_path checkpoints/train_coco/coco_generated_captions_meacap.json
```

**对比指标**：
- 比较 CIDEr、BLEU@4、METEOR、SPICE 等指标
- 分析 MeaCap 模块对性能的影响

---

## 六、代码结构解析

### 6.1 核心替换点

**原始代码**（`validation.py:136-138`）：
```python
logits = image_text_simiarlity(texts_embeddings, temperature=args.temperature, images_features=image_features)
detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
detected_objects = detected_objects[0]
```

**MeaCap 版本**（`validation_meacap.py:67-88`）：
```python
# Retrieve: Find top-K similar memory captions
clip_score, _ = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
    batch_image_embeds, memory_clip_embeddings
)
select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
select_memory_captions = [memory_captions[id] for id in select_memory_ids]

# Filter: Extract key concepts
detected_objects = retrieve_concepts(
    parser_model=parser_model,
    parser_tokenizer=parser_tokenizer,
    wte_model=wte_model,
    select_memory_captions=select_memory_captions,
    image_embeds=batch_image_embeds,
    device=device
)
```

### 6.2 保持不变的部分

- ✅ 软提示生成（`model.mapping_network`）
- ✅ 提示组合（`compose_discrete_prompts`）
- ✅ 文本生成（`beam_search` / `greedy_search`）
- ✅ 结果保存格式

---

## 七、常见问题

### 7.1 记忆库文件不存在

**错误**：
```
FileNotFoundError: Memory caption file not found: data/memory/coco/memory_captions.json
```

**解决方法**：
1. 检查记忆库文件是否存在
2. 确认 `--memory_id` 和 `--memory_caption_path` 参数正确
3. 参考 MeaCap 项目准备记忆库文件

### 7.2 MeaCap 模块导入错误

**错误**：
```
ImportError: cannot import name 'retrieve_concepts' from 'utils.detect_utils'
```

**解决方法**：
1. 确认 `utils/detect_utils.py` 文件存在
2. 确认 `models/clip_utils.py` 文件存在
3. 检查 `utils/__init__.py` 和 `models/__init__.py` 是否正确配置

### 7.3 显存不足

**错误**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：
1. 使用 `--using_image_features` 参数（使用预提取的图像特征）
2. 减小 `--memory_caption_num`（减少检索数量）
3. 对于大型记忆库（CC3M/SS1M），代码会自动使用 CPU 进行检索

---

## 八、性能优化建议

### 8.1 使用预提取图像特征

```bash
python validation_meacap.py \
    --using_image_features \
    --path_of_val_datasets ./annotations/coco/test_captions_ViTB32.pickle \
    ...
```

**优势**：
- 避免重复编码图像
- 减少 GPU 显存占用
- 加快评估速度

### 8.2 调整记忆库检索数量

```bash
# 减少检索数量（更快，但可能降低质量）
--memory_caption_num 3

# 增加检索数量（更慢，但可能提高质量）
--memory_caption_num 10
```

### 8.3 使用离线模式

```bash
# 如果模型已下载到本地
python validation_meacap.py \
    --offline_mode \
    --parser_checkpoint ./checkpoints/flan-t5-base-VG-factual-sg \
    --wte_model_path ./checkpoints/all-MiniLM-L6-v2 \
    ...
```

---

## 九、总结

### 9.1 使用流程

1. **准备记忆库文件**：确保 `data/memory/{memory_id}/` 目录下有必需文件
2. **运行评估**：使用 `validation_meacap.py` 或 `eval_coco_meacap.sh`
3. **查看结果**：检查 `{out_path}/{name_of_datasets}_generated_captions_meacap.json`
4. **计算指标**：使用 `cocoeval.py` 计算评估指标
5. **对比分析**：与原始版本结果对比

### 9.2 关键优势

- ✅ **Plug-and-Play**：无需修改 ViECap 其他部分
- ✅ **结果可对比**：输出文件带 `_meacap` 后缀，便于对比
- ✅ **接口兼容**：参数与原始版本基本一致
- ✅ **灵活配置**：支持多种记忆库和模型配置

### 9.3 下一步

- 运行评估并对比性能指标
- 分析 MeaCap 模块对生成质量的影响
- 根据结果调整参数（`memory_caption_num`、`soft_prompt_first` 等）

