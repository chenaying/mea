# eval_coco.sh 推理代码详细解析

## 一、命令解析

```bash
bash eval_coco.sh train_coco 0 '' 14
```

### 参数说明

| 参数位置 | 值 | 说明 |
|---------|-----|------|
| `$1` | `train_coco` | 实验名称（EXP_NAME） |
| `$2` | `0` | GPU 设备编号（DEVICE） |
| `$3` | `''` | 其他额外参数（OTHER_ARGS） |
| `$4` | `14` | 模型权重对应的 epoch 编号（EPOCH） |

## 二、脚本执行流程

### 2.1 脚本路径设置（eval_coco.sh:1-2）

```bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..
```

- 获取脚本所在目录的绝对路径
- 切换到项目根目录（`scripts/` 的上一级）

### 2.2 参数解析（eval_coco.sh:4-9）

```bash
EXP_NAME=$1           # train_coco
DEVICE=$2             # 0
OTHER_ARGS=$3         # '' (空)
EPOCH=$4              # 14
WEIGHT_PATH=checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt
COCO_OUT_PATH=checkpoints/$EXP_NAME
```

**生成的路径**：
- `WEIGHT_PATH`: `checkpoints/train_coco/coco_prefix-0014.pt`
- `COCO_OUT_PATH`: `checkpoints/train_coco`

### 2.3 日志设置（eval_coco.sh:11-15）

```bash
TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER
COCO_LOG_FILE="$LOG_FOLDER/COCO_${TIME_START}.log"
```

- 创建日志目录：`logs/train_coco_EVAL/`
- 日志文件：`logs/train_coco_EVAL/COCO_2024-01-01-12-00-00.log`（时间戳会变化）

### 2.4 执行验证（eval_coco.sh:17-36）

```bash
python validation.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--top_k 3 \
--threshold 0.4 \
--using_image_features \
--name_of_datasets coco \
--path_of_val_datasets ./annotations/coco/test_captions.json \
--name_of_entities_text coco_entities \
--image_folder ./annotations/coco/val2014/ \
--prompt_ensemble \
--weight_path=$WEIGHT_PATH \
--out_path=$COCO_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
$OTHER_ARGS \
||& tee -a  ${COCO_LOG_FILE}
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--device` | `cuda:0` | 使用 GPU 0 |
| `--clip_model` | `ViT-B/32` | CLIP 模型版本 |
| `--language_model` | `gpt2` | 语言模型（GPT-2） |
| `--continuous_prompt_length` | `10` | 软提示长度（连续嵌入的 token 数） |
| `--top_k` | `3` | 检索 Top-3 实体 |
| `--threshold` | `0.4` | 实体相似度阈值 |
| `--using_image_features` | ✓ | 使用预提取的图像特征（加快推理） |
| `--name_of_datasets` | `coco` | 数据集名称 |
| `--path_of_val_datasets` | `./annotations/coco/test_captions.json` | 测试集标注文件 |
| `--name_of_entities_text` | `coco_entities` | 使用 COCO 实体词汇表 |
| `--prompt_ensemble` | ✓ | 使用 prompt ensemble（多个 prompt 模板的嵌入平均） |
| `--using_hard_prompt` | ✓ | 使用硬提示（离散概念） |
| `--soft_prompt_first` | ✓ | **软提示在前，硬提示在后** |

**注意**：`||& tee -a ${COCO_LOG_FILE}` 表示将标准输出和标准错误都追加到日志文件中。

### 2.5 执行评估（eval_coco.sh:38-39）

```bash
echo "==========================COCO EVAL================================"
python evaluation/cocoeval.py --result_file_path $COCO_OUT_PATH/coco*.json |& tee -a ${COCO_LOG_FILE}
```

- 使用 COCO 评估脚本计算指标（CIDEr、BLEU@4、METEOR、SPICE 等）
- 评估结果追加到日志文件

## 三、validation.py 执行流程

### 3.1 主函数初始化（validation.py:173-191）

```python
def main(args) -> None:
    device = args.device
    clip_name = args.clip_model.replace('/', '')  # 'ViT-B32'
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512
    
    # 加载实体词汇表和嵌入
    if args.name_of_entities_text == 'coco_entities':
        entities_text = load_entities_text(...)
        if args.prompt_ensemble:
            texts_embeddings = clip_texts_embeddings(entities_text, 
                f'./annotations/vocabulary/coco_embeddings_{clip_name}_with_ensemble.pickle')
```

**加载的内容**：
1. **实体词汇表**（`entities_text`）：COCO 类别列表，如 `['person', 'bicycle', 'car', ...]`
2. **实体嵌入**（`texts_embeddings`）：每个实体对应的 CLIP 文本嵌入

### 3.2 模型加载（validation.py:194-207）

```python
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.language_model)

# 加载模型
model = ClipCaptionModel(
    args.continuous_prompt_length,      # 10
    args.clip_project_length,           # 10
    clip_hidden_size,                   # 512
    gpt_type = args.language_model      # 'gpt2'
)
model.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict = False)
model.to(device)
model.eval()

# 加载 CLIP 编码器（如果使用图像而非预提取特征）
if not args.using_image_features:
    encoder, preprocess = clip.load(args.clip_model, device = device)
```

### 3.3 调用验证函数（validation.py:209-235）

根据数据集类型调用不同的验证函数：

```python
if args.name_of_datasets == 'coco':
    validation_coco(args, inpath, entities_text, texts_embeddings, 
                    model, tokenizer, preprocess, encoder)
```

## 四、validation_coco 函数详细解析

### 4.1 数据加载（validation.py:102-118）

```python
def validation_coco(args, inpath, entities_text, texts_embeddings, 
                    model, tokenizer, preprocess, encoder):
    
    if args.using_image_features:
        with open(inpath, 'rb') as infile:
            annotations = pickle.load(infile)
        # annotations: [[image_id, image_features, [caption1, caption2, ...]], ...]
    else:
        with open(inpath, 'r') as infile:
            annotations = json.load(inpath)
        # annotations: [{'image_id': 'xxx.jpg', 'caption': [...]}, ...]
```

**数据格式**：
- 使用预提取特征：`[[image_id, image_features, captions], ...]`
- 使用图像：`[{'image_id': 'xxx.jpg', 'caption': [...]}, ...]`

### 4.2 推理循环（validation.py:120-158）

对每张图像执行以下步骤：

#### 步骤 1：图像特征提取

```python
if args.using_image_features:
    image_id, image_features, captions = annotation
    image_features = image_features.float().unsqueeze(dim = 0).to(device)
else:
    image_id = annotation['image_id']
    captions = annotation['caption']
    image_path = args.image_folder + image_id
    image = preprocess(Image.open(image_path)).unsqueeze(dim = 0).to(device)
    image_features = encoder.encode_image(image).float()

image_features /= image_features.norm(2, dim = -1, keepdim = True)
```

#### 步骤 2：生成软提示

```python
continuous_embeddings = model.mapping_network(image_features).view(
    -1, args.continuous_prompt_length, model.gpt_hidden_size
)
# shape: (1, 10, 768) - 10 个连续嵌入，每个 768 维
```

#### 步骤 3：生成硬提示（如果使用）

```python
if args.using_hard_prompt:
    # 计算图像与实体词汇表的相似度
    logits = image_text_simiarlity(
        texts_embeddings,                # (num_entities, 512)
        temperature = args.temperature,  # 0.01
        images_features = image_features # (1, 512)
    )
    
    # Top-K 检索
    detected_objects, _ = top_k_categories(
        entities_text,      # 实体列表
        logits,             # 相似度分数
        args.top_k,         # 3
        args.threshold      # 0.4
    )
    detected_objects = detected_objects[0]  # ['person', 'bicycle', 'car']
    
    # 构建硬提示
    discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects)
    # "There are person, bicycle, car in image." -> [token_ids]
    discrete_embeddings = model.word_embed(discrete_tokens)
    # shape: (1, discrete_length, 768)
```

#### 步骤 4：组合软提示和硬提示

```python
if args.only_hard_prompt:
    embeddings = discrete_embeddings
elif args.soft_prompt_first:  # ← 当前配置
    embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
    # [软提示(10) + 硬提示(discrete_length)] -> (1, 10+discrete_length, 768)
else:
    embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
    # [硬提示(discrete_length) + 软提示(10)]
```

**关键**：当前配置使用 `--soft_prompt_first`，所以软提示在前，硬提示在后。

#### 步骤 5：文本生成

```python
if 'gpt' in args.language_model:
    if not args.using_greedy_search:
        sentence = beam_search(
            embeddings = embeddings,
            tokenizer = tokenizer,
            beam_width = args.beam_width,  # 默认 5
            model = model.gpt
        )
        sentence = sentence[0]  # 选择 Top-1
    else:
        sentence = greedy_search(...)
```

**Beam Search 说明**：
- 使用束搜索生成文本
- `beam_width=5`：保持 5 个候选序列
- 返回概率最高的序列

#### 步骤 6：保存结果

```python
predict = {
    "image_id": image_id,
    "captions": captions,           # 真实标注（5 个）
    "prediction": sentence          # 模型生成的描述
}
predictions.append(predict)
```

### 4.3 保存预测结果（validation.py:160-162）

```python
with open(os.path.join(args.out_path, 'coco_generated_captions.json'), 'w') as outfile:
    json.dump(predictions, outfile, indent = 4)
```

保存到：`checkpoints/train_coco/coco_generated_captions.json`

## 五、COCO 评估流程

### 5.1 评估脚本（evaluation/cocoeval.py）

```bash
python evaluation/cocoeval.py --result_file_path checkpoints/train_coco/coco*.json
```

**评估指标**：
- **CIDEr**：基于共识的图像描述评估
- **BLEU@4**：n-gram 精确匹配
- **METEOR**：基于对齐的评估
- **SPICE**：基于语义命题的评估

**输出示例**：
```
CIDEr: 115.2
BLEU@4: 36.5
METEOR: 28.3
SPICE: 21.4
```

## 六、完整流程图

```
eval_coco.sh train_coco 0 '' 14
    ↓
[1] 解析参数
    ├─ EXP_NAME = train_coco
    ├─ DEVICE = 0
    ├─ EPOCH = 14
    └─ WEIGHT_PATH = checkpoints/train_coco/coco_prefix-0014.pt
    ↓
[2] 创建日志目录
    └─ logs/train_coco_EVAL/
    ↓
[3] 调用 validation.py
    ↓
    [3.1] 加载实体词汇表和嵌入
    ├─ entities_text: ['person', 'bicycle', ...]
    └─ texts_embeddings: (num_entities, 512)
    ↓
    [3.2] 加载模型
    ├─ ClipCaptionModel
    ├─ GPT-2 tokenizer
    └─ CLIP encoder (如果使用图像)
    ↓
    [3.3] 遍历测试集图像
    ├─ 提取图像特征
    ├─ 生成软提示 (Mapping Network)
    ├─ 检索 Top-K 实体 (硬提示)
    ├─ 组合软提示 + 硬提示 (soft_prompt_first)
    ├─ Beam Search 生成描述
    └─ 保存预测结果
    ↓
[4] 保存结果到
    └─ checkpoints/train_coco/coco_generated_captions.json
    ↓
[5] 执行 COCO 评估
    └─ 计算 CIDEr, BLEU@4, METEOR, SPICE
    ↓
[6] 输出评估指标到日志文件
```

## 七、关键配置说明

### 7.1 软提示在前（--soft_prompt_first）

**配置**：`--soft_prompt_first`

**影响**：
- 软提示（连续嵌入）在前
- 硬提示（离散概念）在后
- 格式：`[软提示(10 tokens)] + [硬提示(discrete_length tokens)]`

**与训练时的对应**：
- 如果训练时使用了 `--soft_prompt_first`，推理时也应该使用
- 保持训练和推理的一致性

### 7.2 Prompt Ensemble（--prompt_ensemble）

**配置**：`--prompt_ensemble`

**影响**：
- 使用多个 prompt 模板的嵌入平均
- 文件：`coco_embeddings_ViT-B32_with_ensemble.pickle`
- 可以提高实体检索的鲁棒性

### 7.3 使用预提取特征（--using_image_features）

**配置**：`--using_image_features`

**影响**：
- 使用预提取的图像特征，跳过 CLIP 编码步骤
- 加快推理速度
- 需要预先提取特征并保存为 pickle 文件

## 八、输出文件

### 8.1 预测结果

**文件**：`checkpoints/train_coco/coco_generated_captions.json`

**格式**：
```json
[
    {
        "image_id": "COCO_val2014_000000000042.jpg",
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

### 8.2 日志文件

**文件**：`logs/train_coco_EVAL/COCO_YYYY-MM-DD-HH-MM-SS.log`

**内容**：
- 模型加载信息
- 推理进度
- 评估指标（CIDEr、BLEU@4、METEOR、SPICE）

## 九、总结

**命令**：`bash eval_coco.sh train_coco 0 '' 14`

**执行内容**：
1. 加载 epoch 14 的模型权重
2. 在 COCO 测试集上推理
3. 使用软提示+硬提示（软提示在前）
4. 生成描述并保存
5. 计算评估指标

**关键特点**：
- ✅ 使用预提取图像特征（快速）
- ✅ 使用 prompt ensemble（鲁棒）
- ✅ 使用软提示+硬提示（性能好）
- ✅ 软提示在前（与训练配置匹配）
- ✅ Beam Search 生成（质量高）

