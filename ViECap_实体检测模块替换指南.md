# ViECap 实体检测模块替换指南

## 一、ViECap 核心代码文件

### 1.1 主要推理文件

| 文件 | 用途 | 实体检测位置 |
|------|------|------------|
| **`infer_by_instance.py`** | 单图推理 | 第67-69行 |
| **`validation.py`** | 模型评估 | 第136-138行（COCO/Flickr30k）<br>第55-57行（NoCaps） |
| **`infer_by_batch.py`** | 批量推理 | 第79-80行 |

### 1.2 核心实体检测模块

**文件**：`retrieval_categories.py`

**关键函数**：
- `image_text_simiarlity()`：计算图像-文本相似度（第61-96行）
- `top_k_categories()`：Top-K实体检索（第98-117行）
- `clip_texts_embeddings()`：预定义实体嵌入（第9-59行）

### 1.3 需要修改的文件

需要替换实体检测模块的文件：
1. ✅ `infer_by_instance.py` - **主要推理文件**
2. ✅ `validation.py` - **评估文件**
3. ✅ `infer_by_batch.py` - **批量推理文件**

## 二、原始实体检测代码分析

### 2.1 核心代码位置

#### 文件1：`infer_by_instance.py`

```python
# 第10行：导入
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories

# 第19-52行：加载预定义实体词汇表
if args.name_of_entities_text == 'coco_entities':
    entities_text = load_entities_text(...)
    texts_embeddings = clip_texts_embeddings(entities_text, ...)

# 第66-70行：实体检测（需要替换的部分）
if args.using_hard_prompt:
    logits = image_text_simiarlity(
        texts_embeddings, 
        temperature = args.temperature, 
        images_features = image_features
    )
    detected_objects, _ = top_k_categories(
        entities_text, 
        logits, 
        args.top_k, 
        args.threshold
    )
    detected_objects = detected_objects[0]  # 单图推理
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects)
```

#### 文件2：`validation.py`

```python
# 第15行：导入
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories

# 第135-139行：实体检测（需要替换）
if args.using_hard_prompt:
    logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
    detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
    detected_objects = detected_objects[0]
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)
```

#### 文件3：`infer_by_batch.py`

```python
# 第13行：导入
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories

# 第79-80行：实体检测（需要替换）
logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
```

### 2.2 原始实现流程

```
1. 加载预定义实体词汇表
   ↓
2. 计算图像与实体的CLIP相似度
   ↓
3. Top-K检索 + 阈值过滤
   ↓
4. 输出概念列表
```

## 三、详细替换步骤

### 步骤1：准备 MeaCap 的 Retrieve-then-Filter 模块

#### 1.1 复制必要文件

从 `MeaCap-main` 复制以下文件到 `ViECap`：

```bash
# 需要复制的文件
MeaCap-main/utils/detect_utils.py → ViECap/utils/detect_utils.py
MeaCap-main/utils/parse_tool_new.py → ViECap/utils/parse_tool_new.py
MeaCap-main/models/clip_utils.py → ViECap/models/clip_utils.py（如果不存在）
```

#### 1.2 安装额外依赖

```bash
pip install sentence-transformers
pip install transformers  # 如果还没有
```

#### 1.3 准备记忆库

```bash
# 下载或预处理记忆库
# 记忆库应放在：data/memory/{memory_id}/
data/memory/coco/
├── memory_captions.json
├── memory_clip_embeddings.pt
└── memory_wte_embeddings.pt
```

### 步骤2：修改 `infer_by_instance.py`

#### 2.1 修改导入部分

**原代码**（第1-10行）：
```python
import clip
import torch
import argparse
from PIL import Image
from ClipCap import ClipCaptionModel
from transformers import AutoTokenizer
from utils import compose_discrete_prompts
from load_annotations import load_entities_text
from search import greedy_search, beam_search, opt_search
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories
```

**修改为**：
```python
import clip
import torch
import argparse
import json
import os
from PIL import Image
from ClipCap import ClipCaptionModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utils import compose_discrete_prompts
from load_annotations import load_entities_text
from search import greedy_search, beam_search, opt_search
# 保留原始导入作为备选
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories
# 添加MeaCap模块导入
from utils.detect_utils import retrieve_concepts
from models.clip_utils import CLIP  # 如果使用MeaCap的CLIP工具类
```

#### 2.2 添加参数

**在参数解析部分添加**（第96-115行后）：
```python
parser.add_argument('--use_memory', action='store_true', default=False, 
                    help='use MeaCap retrieve-then-filter module')
parser.add_argument('--memory_id', type=str, default='coco', 
                    help='memory bank ID')
parser.add_argument('--memory_caption_num', type=int, default=5, 
                    help='number of memory captions to retrieve')
parser.add_argument('--parser_checkpoint', type=str, 
                    default='lizhuang144/flan-t5-base-VG-factual-sg',
                    help='scene graph parser checkpoint')
parser.add_argument('--wte_model_path', type=str, 
                    default='sentence-transformers/all-MiniLM-L6-v2',
                    help='SentenceBERT model path')
parser.add_argument('--vl_model', type=str, 
                    default='openai/clip-vit-base-patch32',
                    help='CLIP model for memory retrieval')
```

#### 2.3 修改主函数 - 添加记忆库初始化

**在 `main()` 函数开始处添加**（第14行后）：
```python
@torch.no_grad()
def main(args) -> None:
    # initializing
    device = args.device
    clip_name = args.clip_model.replace('/', '') 
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    # ========== MeaCap 记忆库初始化（新增） ==========
    if args.use_memory:
        # 1. 加载SceneGraphParser
        parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
        parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint)
        parser_model.eval()
        parser_model.to(device)
        
        # 2. 加载SentenceBERT
        wte_model = SentenceTransformer(args.wte_model_path)
        
        # 3. 加载CLIP（用于记忆库检索）
        vl_model = CLIP(args.vl_model)
        vl_model.device_convert(device)
        
        # 4. 加载记忆库
        memory_id = args.memory_id
        memory_caption_path = os.path.join(f"data/memory/{memory_id}", "memory_captions.json")
        memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_clip_embeddings.pt")
        memory_wte_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_wte_embeddings.pt")
        
        memory_clip_embeddings = torch.load(memory_clip_embedding_file, map_location='cpu')
        memory_wte_embeddings = torch.load(memory_wte_embedding_file, map_location='cpu')
        with open(memory_caption_path, 'r') as f:
            memory_captions = json.load(f)
        
        print(f'Memory bank loaded: {memory_id}')
        print(f'Memory size: {len(memory_captions)} captions')
    else:
        parser_model = None
        parser_tokenizer = None
        wte_model = None
        vl_model = None
        memory_clip_embeddings = None
        memory_captions = None
    # ========== 记忆库初始化结束 ==========

    # 原有的实体词汇表加载（作为备选）
    if not args.use_memory:
        # ... 原有的加载代码 ...
```

#### 2.4 替换实体检测部分

**原代码**（第66-70行）：
```python
if args.using_hard_prompt:
    logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
    detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
    detected_objects = detected_objects[0]
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)
```

**替换为**：
```python
if args.using_hard_prompt:
    if args.use_memory:
        # ========== MeaCap Retrieve-then-Filter 方法 ==========
        # 1. 检索记忆库
        batch_image_embeds = vl_model.compute_image_representation_from_image_path(args.image_path)
        clip_score, clip_ref = vl_model.compute_image_text_similarity_via_embeddings(
            batch_image_embeds, 
            memory_clip_embeddings.to(device) if memory_clip_embeddings.device != device else memory_clip_embeddings
        )
        select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
        select_memory_captions = [memory_captions[id] for id in select_memory_ids]
        
        # 2. Retrieve-then-Filter提取概念
        detected_objects = retrieve_concepts(
            parser_model=parser_model,
            parser_tokenizer=parser_tokenizer,
            wte_model=wte_model,
            select_memory_captions=select_memory_captions,
            image_embeds=batch_image_embeds,
            device=device,
            logger=None  # 可以传入logger用于日志
        )
        print(f"MeaCap detected concepts: {detected_objects}")
    else:
        # ========== ViECap 原始方法（备选） ==========
        logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
        detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
        detected_objects = detected_objects[0]
        print(f"ViECap detected concepts: {detected_objects}")
    
    # 后续处理完全相同
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)
```

### 步骤3：修改 `validation.py`

#### 3.1 修改导入部分

**在文件开头添加**：
```python
# 原有导入...
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories

# 添加MeaCap导入
import json
import os
from transformers import AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utils.detect_utils import retrieve_concepts
from models.clip_utils import CLIP
```

#### 3.2 修改 `main()` 函数

**在函数开始处添加记忆库初始化**（类似 `infer_by_instance.py`）

#### 3.3 替换实体检测部分

**找到两处实体检测代码并替换**：

**位置1**：`validation_coco_flickr30k()` 函数（第135-139行）

**原代码**：
```python
if args.using_hard_prompt:
    logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
    detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
    detected_objects = detected_objects[0]
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)
```

**替换为**：
```python
if args.using_hard_prompt:
    if args.use_memory:
        # MeaCap方法
        clip_score, clip_ref = vl_model.compute_image_text_similarity_via_embeddings(
            image_features, 
            memory_clip_embeddings.to(device)
        )
        select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
        select_memory_captions = [memory_captions[id] for id in select_memory_ids]
        
        detected_objects = retrieve_concepts(
            parser_model=parser_model,
            parser_tokenizer=parser_tokenizer,
            wte_model=wte_model,
            select_memory_captions=select_memory_captions,
            image_embeds=image_features,
            device=device
        )
    else:
        # ViECap原始方法
        logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
        detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
        detected_objects = detected_objects[0]
    
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)
```

**位置2**：`validation_nocaps()` 函数（第54-58行）- 同样替换

### 步骤4：修改 `infer_by_batch.py`

#### 4.1 修改导入和初始化（类似步骤2）

#### 4.2 替换批量推理中的实体检测

**原代码**（第79-80行）：
```python
logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
```

**替换为**（在循环中）：
```python
if args.use_memory:
    # MeaCap方法
    clip_score, clip_ref = vl_model.compute_image_text_similarity_via_embeddings(
        image_features, 
        memory_clip_embeddings.to(device)
    )
    select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
    select_memory_captions = [memory_captions[id] for id in select_memory_ids]
    
    detected_objects = retrieve_concepts(
        parser_model=parser_model,
        parser_tokenizer=parser_tokenizer,
        wte_model=wte_model,
        select_memory_captions=select_memory_captions,
        image_embeds=image_features,
        device=device
    )
    detected_objects = [detected_objects]  # 批量格式
else:
    # ViECap原始方法
    logits = image_text_simiarlity(texts_embeddings, temperature = args.temperature, images_features = image_features)
    detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
```

## 四、完整代码示例

### 4.1 修改后的 `infer_by_instance.py` 关键部分

```python
@torch.no_grad()
def main(args) -> None:
    device = args.device
    clip_name = args.clip_model.replace('/', '') 
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    # ========== MeaCap 记忆库初始化 ==========
    if args.use_memory:
        parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
        parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint)
        parser_model.eval().to(device)
        
        wte_model = SentenceTransformer(args.wte_model_path)
        
        vl_model = CLIP(args.vl_model)
        vl_model.device_convert(device)
        
        memory_id = args.memory_id
        memory_caption_path = f"data/memory/{memory_id}/memory_captions.json"
        memory_clip_embedding_file = f"data/memory/{memory_id}/memory_clip_embeddings.pt"
        
        memory_clip_embeddings = torch.load(memory_clip_embedding_file, map_location='cpu')
        with open(memory_caption_path, 'r') as f:
            memory_captions = json.load(f)
    else:
        # 原有实体词汇表加载
        entities_text = load_entities_text(...)
        texts_embeddings = clip_texts_embeddings(...)
        parser_model = None
        parser_tokenizer = None
        wte_model = None
        vl_model = None

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(...)
    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
    model.to(device)
    encoder, preprocess = clip.load(args.clip_model, device=device)
   
    # 图像编码
    image = preprocess(Image.open(args.image_path)).unsqueeze(dim=0).to(device)
    image_features = encoder.encode_image(image).float()
    image_features /= image_features.norm(2, dim=-1, keepdim=True)
    continuous_embeddings = model.mapping_network(image_features).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
    
    # ========== 实体检测（替换部分） ==========
    if args.using_hard_prompt:
        if args.use_memory:
            # MeaCap Retrieve-then-Filter
            batch_image_embeds = vl_model.compute_image_representation_from_image_path(args.image_path)
            clip_score, _ = vl_model.compute_image_text_similarity_via_embeddings(
                batch_image_embeds, 
                memory_clip_embeddings.to(device)
            )
            select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
            select_memory_captions = [memory_captions[id] for id in select_memory_ids]
            
            detected_objects = retrieve_concepts(
                parser_model=parser_model,
                parser_tokenizer=parser_tokenizer,
                wte_model=wte_model,
                select_memory_captions=select_memory_captions,
                image_embeds=batch_image_embeds,
                device=device
            )
        else:
            # ViECap 原始方法
            logits = image_text_simiarlity(texts_embeddings, temperature=args.temperature, images_features=image_features)
            detected_objects, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
            detected_objects = detected_objects[0]
        
        # 后续处理完全相同
        discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(device)
        discrete_embeddings = model.word_embed(discrete_tokens)
        
        if args.only_hard_prompt:
            embeddings = discrete_embeddings
        elif args.soft_prompt_first:
            embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
        else:
            embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)
    else:
        embeddings = continuous_embeddings
    
    # 文本生成（完全不变）
    if 'gpt' in args.language_model:
        if not args.using_greedy_search:
            sentence = beam_search(embeddings=embeddings, tokenizer=tokenizer, beam_width=args.beam_width, model=model.gpt)[0]
        else:
            sentence = greedy_search(embeddings=embeddings, tokenizer=tokenizer, model=model.gpt)
    
    print(f'the generated caption: {sentence}')
```

## 五、替换后的使用方式

### 5.1 使用 MeaCap 方法

```bash
python infer_by_instance.py \
    --use_memory \
    --memory_id coco \
    --memory_caption_num 5 \
    --using_hard_prompt \
    --soft_prompt_first \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt
```

### 5.2 使用 ViECap 原始方法

```bash
python infer_by_instance.py \
    --using_hard_prompt \
    --soft_prompt_first \
    --image_path ./images/instance1.jpg \
    --weight_path ./checkpoints/train_coco/coco_prefix-0014.pt
```

## 六、关键注意事项

### 6.1 设备管理

```python
# 大记忆库（CC3M/SS1M）可能需要CPU检索
if memory_id in ['cc3m', 'ss1m']:
    memory_clip_embeddings = memory_clip_embeddings.to('cpu')
    # 在CPU上计算相似度
```

### 6.2 内存优化

```python
# 如果显存不足，可以分批处理
batch_size = 32
for i in range(0, len(memory_captions), batch_size):
    batch_embeddings = memory_clip_embeddings[i:i+batch_size]
    # 处理批次
```

### 6.3 错误处理

```python
try:
    detected_objects = retrieve_concepts(...)
except Exception as e:
    print(f"MeaCap retrieval failed: {e}")
    # 回退到原始方法
    logits = image_text_simiarlity(...)
    detected_objects, _ = top_k_categories(...)
```

## 七、验证替换

### 7.1 测试脚本

```python
# test_replacement.py
import torch
from infer_by_instance import main
import argparse

args = argparse.Namespace(
    device='cuda:0',
    clip_model='ViT-B/32',
    language_model='gpt2',
    continuous_prompt_length=10,
    clip_project_length=10,
    using_hard_prompt=True,
    soft_prompt_first=True,
    image_path='./images/instance1.jpg',
    weight_path='./checkpoints/train_coco/coco_prefix-0014.pt',
    use_memory=True,  # 测试MeaCap方法
    memory_id='coco',
    memory_caption_num=5,
    parser_checkpoint='lizhuang144/flan-t5-base-VG-factual-sg',
    wte_model_path='sentence-transformers/all-MiniLM-L6-v2',
    vl_model='openai/clip-vit-base-patch32',
    using_greedy_search=False,
    beam_width=5
)

main(args)
```

### 7.2 对比测试

```bash
# 测试原始方法
python infer_by_instance.py --image_path test.jpg --weight_path model.pt

# 测试MeaCap方法
python infer_by_instance.py --use_memory --memory_id coco --image_path test.jpg --weight_path model.pt

# 对比输出结果
```

## 八、总结

### 8.1 核心文件

- **主要推理**：`infer_by_instance.py`
- **评估**：`validation.py`
- **批量推理**：`infer_by_batch.py`
- **实体检测模块**：`retrieval_categories.py`

### 8.2 替换要点

1. ✅ **只需替换实体检测部分**（3-5行代码）
2. ✅ **添加记忆库初始化**（一次性设置）
3. ✅ **后续处理完全不变**
4. ✅ **支持两种方法切换**（通过 `--use_memory` 参数）

### 8.3 替换成本

- **代码修改**：每个文件约10-20行
- **依赖添加**：SceneGraphParser + SentenceBERT
- **数据准备**：记忆库预处理
- **性能影响**：推理时间增加约20-30%（检索+解析）

### 8.4 预期效果

根据论文，替换后性能提升：
- COCO: 92.9 → **95.4** CIDEr (+2.5)
- Flickr30k: 47.9 → **59.4** CIDEr (+11.5)

---

**关键提示**：替换时确保：
1. ✅ 记忆库文件路径正确
2. ✅ 所有依赖已安装
3. ✅ 设备内存充足（大记忆库可能需要CPU检索）
4. ✅ 保留原始方法作为备选


