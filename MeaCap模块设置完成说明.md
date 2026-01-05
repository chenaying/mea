# MeaCap模块设置完成说明

## ✅ 已完成的设置

### 1. 文件复制

已成功复制以下文件：

- ✅ `utils/detect_utils.py` - MeaCap的核心检索函数
- ✅ `utils/parse_tool.py` - 场景图解析工具
- ✅ `models/clip_utils.py` - CLIP工具类
- ✅ `models/__init__.py` - models包初始化文件
- ✅ `utils/__init__.py` - 已更新，支持从utils.py导入函数

### 2. 代码修改

- ✅ `infer_by_instance_modified.py` - 已集成MeaCap模块
- ✅ `validation_modified.py` - 已集成MeaCap模块
- ✅ `infer_by_batch_modified.py` - 已集成MeaCap模块

## ⚠️ 还需要完成的设置

### 1. 安装依赖

```bash
pip install sentence-transformers
pip install nltk  # parse_tool.py需要
```

### 2. 下载NLTK数据

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 3. 准备记忆库文件

记忆库文件需要放在以下位置：

```
data/memory/{memory_id}/
├── memory_captions.json          # 记忆描述列表
├── memory_clip_embeddings.pt     # CLIP嵌入
└── memory_wte_embeddings.pt      # SentenceBERT嵌入（可选）
```

**记忆库ID选项**：
- `coco` - COCO数据集记忆库
- `flickr30k` - Flickr30k数据集记忆库
- `cc3m` - CC3M数据集记忆库（大记忆库）
- `ss1m` - SS1M数据集记忆库（大记忆库）

### 4. 记忆库文件格式

#### `memory_captions.json`
```json
[
  "A young girl is sitting on a bed with a teddy bear.",
  "A person is sitting on a bed.",
  ...
]
```

#### `memory_clip_embeddings.pt`
```python
# PyTorch tensor, shape: (num_captions, clip_embedding_dim)
# 例如: (50000, 512) for COCO
```

#### `memory_wte_embeddings.pt`（可选）
```python
# PyTorch tensor, shape: (num_captions, wte_embedding_dim)
```

## 🚀 使用方法

### 测试导入

```bash
python -c "from utils.detect_utils import retrieve_concepts; from models.clip_utils import CLIP; print('Import successful!')"
```

### 运行推理（使用MeaCap）

```bash
python infer_by_instance_modified.py \
    --use_memory \
    --memory_id coco \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --soft_prompt_first \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt
```

### 运行推理（使用ViECap原始方法）

```bash
python infer_by_instance_modified.py \
    --using_hard_prompt \
    --soft_prompt_first \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt
```

## 📝 注意事项

### 1. 记忆库路径

确保记忆库文件路径正确：
- 默认路径：`data/memory/{memory_id}/`
- 如果路径不同，需要修改代码中的路径

### 2. 大记忆库处理

对于大记忆库（CC3M、SS1M），代码会自动在CPU上检索，避免显存不足。

### 3. 依赖检查

如果MeaCap模块不可用，代码会自动回退到ViECap原始方法，不会报错。

### 4. 错误处理

代码包含完整的错误处理：
- 记忆库文件不存在 → 报错并提示
- 模块导入失败 → 自动回退到原始方法
- 检索失败 → 自动回退到原始方法

## 🔍 验证步骤

### 步骤1：检查文件是否存在

```bash
ls -la utils/detect_utils.py
ls -la utils/parse_tool.py
ls -la models/clip_utils.py
```

### 步骤2：测试导入

```python
python -c "
from utils.detect_utils import retrieve_concepts
from models.clip_utils import CLIP
print('✓ All imports successful!')
"
```

### 步骤3：检查记忆库

```bash
ls -la data/memory/coco/
```

### 步骤4：运行测试

```bash
python infer_by_instance_modified.py \
    --use_memory \
    --memory_id coco \
    --using_hard_prompt \
    --image_path ./images/instance1.jpg
```

## ❓ 常见问题

### Q1: 提示 "MeaCap modules not found"

**原因**：模块文件未正确复制或导入路径错误

**解决**：
1. 检查文件是否存在
2. 检查 `utils/__init__.py` 是否正确
3. 检查Python路径

### Q2: 提示 "Memory caption file not found"

**原因**：记忆库文件不存在

**解决**：
1. 检查记忆库路径
2. 确保文件格式正确
3. 检查 `--memory_id` 参数

### Q3: 导入nltk错误

**原因**：NLTK数据未下载

**解决**：
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

### Q4: 显存不足

**原因**：大记忆库在GPU上检索

**解决**：
- 代码会自动在CPU上检索大记忆库
- 或手动设置 `retrieve_on_CPU = True`

---

**设置完成后，MeaCap模块应该可以正常工作了！** 🎉


