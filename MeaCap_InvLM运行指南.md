# 在 ViECap 中运行 MeaCap InvLM 指南

## 一、概述

**是的，可以直接在 ViECap 模型中运行 `viecap_inference.py` 来实现 MeaCap InvLM！**

我已经创建了适配版本的 `viecap_inference_adapted.py`，它将 MeaCap-main 项目中的 `viecap_inference.py` 适配到 ViECap 项目结构中。

---

## 二、主要修改

### 2.1 导入路径修改

**原 MeaCap-main 版本**（使用 `viecap` 包）：
```python
from viecap.ClipCap import ClipCaptionModel
from viecap.utils import compose_discrete_prompts
from viecap.search import greedy_search, beam_search, opt_search
```

**适配 ViECap 版本**（直接导入）：
```python
from ClipCap import ClipCaptionModel
from utils import compose_discrete_prompts
from search import greedy_search, beam_search, opt_search
```

### 2.2 修复缺失的导入

添加了缺失的导入和变量定义：
```python
import copy  # 用于 copy.deepcopy
cpu_device = torch.device('cpu')  # 用于大型记忆库的 CPU 检索
```

---

## 三、前置条件

### 3.1 必需的 MeaCap 模块文件

**⚠️ 重要**：在运行之前，需要先将以下文件从 `MeaCap-main` 复制到 `ViECap` 项目：

需要复制/创建的文件：

1. **`models/clip_utils.py`** - CLIP 工具类（从 `MeaCap-main/models/clip_utils.py` 复制）
2. **`utils/detect_utils.py`** - Retrieve-then-Filter 核心函数（从 `MeaCap-main/utils/detect_utils.py` 复制）
3. **`utils/parse_tool.py`** - 场景图解析工具（从 `MeaCap-main/utils/parse_tool.py` 复制）
4. **`models/__init__.py`** - models 包初始化（已存在，但需确保为空或正确导出）
5. **`utils/__init__.py`** - utils 包初始化（需导出 `compose_discrete_prompts` 和 MeaCap 相关函数）

**快速复制命令**（在 ViECap 项目根目录执行）：

```bash
# Windows PowerShell
Copy-Item "..\MeaCap-main\models\clip_utils.py" -Destination "models\clip_utils.py" -Force
Copy-Item "..\MeaCap-main\utils\detect_utils.py" -Destination "utils\detect_utils.py" -Force
Copy-Item "..\MeaCap-main\utils\parse_tool.py" -Destination "utils\parse_tool.py" -Force

# Linux/Mac
cp ../MeaCap-main/models/clip_utils.py models/clip_utils.py
cp ../MeaCap-main/utils/detect_utils.py utils/detect_utils.py
cp ../MeaCap-main/utils/parse_tool.py utils/parse_tool.py
```

**验证文件是否存在**：

```bash
# 检查文件
ls models/clip_utils.py
ls utils/detect_utils.py
ls utils/parse_tool.py
```

### 3.2 安装依赖

```bash
pip install sentence-transformers
pip install nltk
```

### 3.3 下载 NLTK 数据

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 3.4 准备记忆库文件

在 `data/memory/coco/` 目录下准备以下文件：

```
data/memory/coco/
├── memory_captions.json          # 记忆库描述文本列表
├── memory_clip_embeddings.pt     # CLIP 嵌入（用于检索）
└── memory_wte_embeddings.pt      # SentenceBERT 嵌入（用于过滤）
```

**生成记忆库文件**：
- 使用 MeaCap-main 项目中的 `prepare_embedding.py` 脚本生成
- 或使用 MeaCap 提供的预训练记忆库

---

## 四、运行方法

### 4.1 基本运行

```bash
python viecap_inference_adapted.py \
    --memory_id coco \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt
```

### 4.2 完整参数示例

```bash
python viecap_inference_adapted.py \
    --device cuda:0 \
    --clip_model ViT-B/32 \
    --language_model openai-community/gpt2 \
    --vl_model openai/clip-vit-base-patch32 \
    --parser_checkpoint lizhuang144/flan-t5-base-VG-factual-sg \
    --wte_model_path sentence-transformers/all-MiniLM-L6-v2 \
    --continuous_prompt_length 10 \
    --clip_project_length 10 \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt \
    --image_path ./images/instance1.jpg \
    --using_hard_prompt \
    --soft_prompt_first \
    --memory_id coco \
    --memory_caption_num 5 \
    --beam_width 5
```

### 4.3 仅使用硬提示

```bash
python viecap_inference_adapted.py \
    --only_hard_prompt \
    --memory_id coco \
    --image_path ./images/instance1.jpg
```

---

## 五、关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--memory_id` | 记忆库名称（coco/cc3m/ss1m） | `coco` |
| `--memory_caption_num` | 检索的记忆描述数量（Top-K） | `5` |
| `--vl_model` | CLIP 模型（用于检索） | `openai/clip-vit-base-patch32` |
| `--parser_checkpoint` | 场景图解析器 | `lizhuang144/flan-t5-base-VG-factual-sg` |
| `--wte_model_path` | SentenceBERT 模型 | `sentence-transformers/all-MiniLM-L6-v2` |
| `--using_hard_prompt` | 是否使用硬提示 | `True` |
| `--soft_prompt_first` | Soft Prompt 是否在前 | `False` |
| `--only_hard_prompt` | 仅使用硬提示 | `False` |

---

## 六、工作流程

```
输入图像
    ↓
[1] 加载 ViECap 模型和 CLIP 编码器
    ↓
[2] 提取图像特征，生成 Soft Prompt
    ↓
[3] 如果使用硬提示 (using_hard_prompt=True)
    ↓
    [3.1] Retrieve 阶段
    ├─ 加载记忆库（memory_captions.json, memory_clip_embeddings.pt）
    ├─ 计算图像与记忆库的 CLIP 相似度
    └─ Top-K 检索最相似的描述
    ↓
    [3.2] Filter 阶段
    ├─ 场景图解析（Flan-T5）
    ├─ 实体提取与合并（SentenceBERT）
    ├─ 图像相关性过滤
    └─ 得到 detected_objects（Top-4 概念）
    ↓
    [3.3] 构建硬提示
    ├─ compose_discrete_prompts → discrete_tokens
    ├─ word_embed → discrete_embeddings
    └─ 组合 Soft + Hard Prompts
    ↓
[4] GPT-2/OPT 生成描述
    ├─ Beam Search / Greedy Search
    └─ 输出最终描述
```

---

## 七、与现有代码的关系

### 7.1 与 `infer_by_instance_modified.py` 的区别

| 文件 | 特点 | 用途 |
|------|------|------|
| `infer_by_instance_modified.py` | 可选使用 MeaCap 模块（通过 `--use_memory` 参数切换） | 保留 ViECap 原始功能，可选启用 MeaCap |
| `viecap_inference_adapted.py` | **直接使用 MeaCap 的 Retrieve-then-Filter** | **完整的 MeaCap InvLM 实现** |

### 7.2 推荐使用场景

- **`viecap_inference_adapted.py`**：当你想要使用完整的 MeaCap InvLM 方法时
- **`infer_by_instance_modified.py`**：当你想要在 ViECap 和 MeaCap 之间切换对比时

---

## 八、常见问题

### 8.1 模块未找到错误

**错误**：
```
ModuleNotFoundError: No module named 'models.clip_utils'
```

**解决**：
- 确保 `models/clip_utils.py` 文件存在
- 确保 `models/__init__.py` 文件存在

### 8.2 记忆库文件未找到

**错误**：
```
FileNotFoundError: data/memory/coco/memory_captions.json
```

**解决**：
- 使用 `prepare_embedding.py` 生成记忆库文件
- 或检查 `--memory_id` 参数是否正确

### 8.3 CUDA 内存不足

**错误**：
```
torch.OutOfMemoryError: CUDA out of memory
```

**解决**：
- 对于大型记忆库（CC3M、SS1M），代码会自动使用 CPU 检索
- 对于小型记忆库，可以减小 `--memory_caption_num` 参数

---

## 九、总结

**答案：是的，可以直接在 ViECap 中运行 `viecap_inference_adapted.py` 来实现 MeaCap InvLM！**

**优势**：
1. ✅ 完整的 MeaCap InvLM 实现
2. ✅ 适配 ViECap 项目结构
3. ✅ 修复了原始代码中的 bug
4. ✅ 支持大型记忆库（CPU 检索优化）

**下一步**：
1. 确保所有必需的模块文件已复制
2. 安装依赖包（sentence-transformers、nltk）
3. 准备记忆库文件
4. 运行 `viecap_inference_adapted.py` 开始推理！

