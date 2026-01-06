# viecap_inference_adapted.py 日志查看指南

## 一、当前输出方式

`viecap_inference_adapted.py` 目前使用 `print()` 函数输出信息，所有输出都显示在**终端（标准输出）**中。

### 1.1 输出内容

运行时会输出以下信息：

1. **参数信息**（第188行）：
   ```python
   print('args: {}\n'.format(vars(args)))
   ```
   输出所有运行参数

2. **模型加载信息**：
   - `Load CLIP from the checkpoint {}.`（第57行）
   - `Load sentenceBERT from the checkpoint {}.`（第70行）
   - `Load Textual Scene Graph parser from the checkpoint {}.`（第85行）

3. **记忆库信息**（如果使用大型记忆库）：
   - `CC3M/SS1M Memory is too big to compute on RTX 3090, Moving to CPU...`（第100行）

4. **提取的概念**（如果使用硬提示）：
   ```python
   print("memory concepts:", detected_objects)  # 第133行
   ```
   例如：`memory concepts: ['cute girl', 'bed']`

5. **生成的描述**（第156行）：
   ```python
   print(f'the generated caption: {sentence}')
   ```
   例如：`the generated caption: A cute girl sitting on a bed with a pink blanket.`

---

## 二、查看日志的方法

### 方法 1：直接查看终端输出（最简单）

运行脚本时，所有输出会直接显示在终端：

```bash
python viecap_inference_adapted.py \
    --language_model ./checkpoints/gpt2 \
    --parser_checkpoint ./checkpoints/flan-t5-base-VG-factual-sg \
    --wte_model_path ./checkpoints/all-MiniLM-L6-v2 \
    --memory_id coco \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt
```

**输出示例**：
```
args: {'device': 'cuda:0', 'clip_model': 'ViT-B/32', ...}

Load CLIP from the checkpoint checkpoints/clip-vit-base-patch32.
Load sentenceBERT from the checkpoint ./checkpoints/all-MiniLM-L6-v2.
Load Textual Scene Graph parser from the checkpoint ./checkpoints/flan-t5-base-VG-factual-sg.
memory concepts: ['cute girl', 'bed']
the generated caption: A cute girl sitting on a bed with a pink blanket.
```

---

### 方法 2：使用重定向保存到文件（推荐）

#### 2.1 保存标准输出

```bash
python viecap_inference_adapted.py \
    --language_model ./checkpoints/gpt2 \
    --parser_checkpoint ./checkpoints/flan-t5-base-VG-factual-sg \
    --wte_model_path ./checkpoints/all-MiniLM-L6-v2 \
    --memory_id coco \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt \
    > inference_log.txt
```

**日志文件**：`inference_log.txt`

#### 2.2 同时显示和保存（tee 命令）

**Linux/Mac**：
```bash
python viecap_inference_adapted.py \
    --language_model ./checkpoints/gpt2 \
    --parser_checkpoint ./checkpoints/flan-t5-base-VG-factual-sg \
    --wte_model_path ./checkpoints/all-MiniLM-L6-v2 \
    --memory_id coco \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt \
    | tee inference_log.txt
```

**Windows PowerShell**：
```powershell
python viecap_inference_adapted.py `
    --language_model ./checkpoints/gpt2 `
    --parser_checkpoint ./checkpoints/flan-t5-base-VG-factual-sg `
    --wte_model_path ./checkpoints/all-MiniLM-L6-v2 `
    --memory_id coco `
    --memory_caption_num 5 `
    --using_hard_prompt `
    --image_path ./images/instance1.jpg `
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt `
    | Tee-Object -FilePath inference_log.txt
```

**效果**：输出同时显示在终端和保存到文件

#### 2.3 保存标准输出和错误输出

```bash
# Linux/Mac
python viecap_inference_adapted.py ... 2>&1 | tee inference_log.txt

# Windows PowerShell
python viecap_inference_adapted.py ... *>&1 | Tee-Object -FilePath inference_log.txt
```

**说明**：
- `2>&1`：将标准错误重定向到标准输出
- `*>&1`：PowerShell 中重定向所有输出

---

### 方法 3：添加时间戳的日志文件

```bash
# Linux/Mac
TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")
python viecap_inference_adapted.py ... 2>&1 | tee logs/inference_${TIMESTAMP}.log

# Windows PowerShell
$timestamp = Get-Date -Format "yyyy-MM-dd-HH-mm-ss"
python viecap_inference_adapted.py ... *>&1 | Tee-Object -FilePath "logs/inference_${timestamp}.log"
```

**日志文件**：`logs/inference_2024-01-01-12-00-00.log`

---

### 方法 4：追加到日志文件

```bash
# Linux/Mac
python viecap_inference_adapted.py ... >> inference_log.txt 2>&1

# Windows PowerShell
python viecap_inference_adapted.py ... *>> inference_log.txt
```

**说明**：`>>` 表示追加，不会覆盖已有内容

---

## 三、查看保存的日志文件

### 3.1 使用文本编辑器

```bash
# Linux/Mac
cat inference_log.txt
# 或
less inference_log.txt
# 或
nano inference_log.txt

# Windows
notepad inference_log.txt
# 或
type inference_log.txt
```

### 3.2 使用 grep 查找特定内容（Linux/Mac）

```bash
# 查找生成的描述
grep "the generated caption" inference_log.txt

# 查找提取的概念
grep "memory concepts" inference_log.txt

# 查找错误信息
grep -i "error\|exception\|traceback" inference_log.txt
```

### 3.3 使用 PowerShell 查找（Windows）

```powershell
# 查找生成的描述
Select-String -Path inference_log.txt -Pattern "the generated caption"

# 查找提取的概念
Select-String -Path inference_log.txt -Pattern "memory concepts"
```

---

## 四、改进建议：添加日志文件保存功能

### 4.1 当前代码的局限性

- 只输出到终端，没有自动保存
- 如果终端关闭，输出会丢失
- 不方便批量处理和后续分析

### 4.2 建议的改进方案

可以修改代码，添加日志文件保存功能：

```python
# 在文件开头添加
import logging
from datetime import datetime

# 在 main 函数开始处添加
def main(args) -> None:
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = f"logs/inference_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到终端
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f'Args: {vars(args)}')
    # ... 其他代码
```

---

## 五、完整的日志查看工作流

### 5.1 运行并保存日志

```bash
# 创建日志目录
mkdir -p logs

# 运行并保存日志
python viecap_inference_adapted.py \
    --language_model ./checkpoints/gpt2 \
    --parser_checkpoint ./checkpoints/flan-t5-base-VG-factual-sg \
    --wte_model_path ./checkpoints/all-MiniLM-L6-v2 \
    --memory_id coco \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt \
    2>&1 | tee logs/inference_$(date +%Y-%m-%d-%H-%M-%S).log
```

### 5.2 查看日志

```bash
# 查看最新日志
ls -t logs/inference_*.log | head -1 | xargs cat

# 查看所有生成的描述
grep "the generated caption" logs/inference_*.log

# 查看所有提取的概念
grep "memory concepts" logs/inference_*.log
```

---

## 六、日志内容示例

### 6.1 完整日志示例

```
args: {'device': 'cuda:0', 'clip_model': 'ViT-B/32', 'language_model': './checkpoints/gpt2', 'vl_model': 'openai/clip-vit-base-patch32', 'parser_checkpoint': './checkpoints/flan-t5-base-VG-factual-sg', 'wte_model_path': './checkpoints/all-MiniLM-L6-v2', 'continuous_prompt_length': 10, 'clip_project_length': 10, 'temperature': 0.01, 'top_k': 3, 'threshold': 0.2, 'disable_all_entities': False, 'name_of_entities_text': 'coco_entities', 'prompt_ensemble': False, 'weight_path': './checkpoints/train_coco/coco_prefix-0014.pt', 'image_path': './images/instance1.jpg', 'using_hard_prompt': True, 'soft_prompt_first': False, 'only_hard_prompt': False, 'using_greedy_search': False, 'beam_width': 5, 'text_prompt': None, 'memory_id': 'coco', 'memory_caption_path': 'data/memory/coco/memory_captions.json', 'memory_caption_num': 5, 'offline_mode': False}

Initializing CLIP model...
CLIP model initialized.
Load CLIP from the checkpoint checkpoints/clip-vit-base-patch32.
Load sentenceBERT from the checkpoint ./checkpoints/all-MiniLM-L6-v2.
Load Textual Scene Graph parser from the checkpoint ./checkpoints/flan-t5-base-VG-factual-sg.
memory concepts: ['cute girl', 'bed']
the generated caption: A cute girl sitting on a bed with a pink blanket.
```

### 6.2 关键信息提取

**参数信息**：
- 设备：`cuda:0`
- 模型路径：`./checkpoints/train_coco/coco_prefix-0014.pt`
- 图像路径：`./images/instance1.jpg`
- 记忆库：`coco`
- 记忆描述数量：`5`

**运行结果**：
- 提取的概念：`['cute girl', 'bed']`
- 生成的描述：`A cute girl sitting on a bed with a pink blanket.`

---

## 七、批量处理时的日志管理

### 7.1 为每张图像创建单独日志

```bash
for image in ./images/*.jpg; do
    image_name=$(basename "$image" .jpg)
    python viecap_inference_adapted.py \
        --image_path "$image" \
        ... \
        2>&1 | tee "logs/${image_name}_$(date +%Y-%m-%d-%H-%M-%S).log"
done
```

### 7.2 汇总所有结果

```bash
# 提取所有生成的描述
grep "the generated caption" logs/*.log > all_predictions.txt

# 提取所有提取的概念
grep "memory concepts" logs/*.log > all_concepts.txt
```

---

## 八、总结

### 8.1 当前方式

- **输出位置**：终端（标准输出）
- **保存方式**：需要手动重定向
- **查看方式**：直接查看终端或查看保存的文件

### 8.2 推荐方式

**方法 1（最简单）**：使用 `tee` 命令
```bash
python viecap_inference_adapted.py ... | tee inference_log.txt
```

**方法 2（带时间戳）**：
```bash
python viecap_inference_adapted.py ... 2>&1 | tee logs/inference_$(date +%Y-%m-%d-%H-%M-%S).log
```

### 8.3 日志文件位置

- **默认**：当前目录下的 `inference_log.txt`
- **推荐**：`logs/inference_YYYY-MM-DD-HH-MM-SS.log`

### 8.4 关键信息

日志中包含的关键信息：
1. ✅ 运行参数（`args`）
2. ✅ 模型加载状态
3. ✅ 提取的概念（`memory concepts`）
4. ✅ 生成的描述（`the generated caption`）

---

## 九、快速参考

### 查看最新日志

```bash
# Linux/Mac
cat inference_log.txt
# 或
tail -f inference_log.txt  # 实时查看

# Windows
type inference_log.txt
# 或
Get-Content inference_log.txt -Wait  # 实时查看
```

### 查找特定内容

```bash
# Linux/Mac
grep "the generated caption" inference_log.txt

# Windows PowerShell
Select-String -Path inference_log.txt -Pattern "the generated caption"
```

