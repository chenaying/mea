# ViECap 评估错误分析与解决方案

## 一、错误概述

评估COCO数据集时遇到两个问题：
1. **CUDA显存不足**：加载模型权重时GPU 0显存被占用
2. **缺少依赖模块**：评估脚本需要`skimage`模块

## 二、错误详细分析

### 2.1 CUDA显存不足错误

```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 148.00 MiB. 
GPU 0 has a total capacity of 10.75 GiB of which 56.75 MiB is free. 
Process 1037648 has 10.54 GiB memory in use.
```

**错误原因**：
1. **GPU 0被其他进程占用**：进程1037648占用了10.54 GiB显存
2. **加载方式问题**：代码直接使用`map_location=device`加载到GPU
3. **显存碎片**：可能存在的显存碎片化问题

**问题代码位置**：
```python
# validation.py:217
model.load_state_dict(torch.load(args.weight_path, map_location = device))
```

### 2.2 缺少skimage模块错误

```
ModuleNotFoundError: No module named 'skimage'
```

**错误原因**：
- `pycocotools`的评估工具需要`scikit-image`库
- 当前环境未安装该依赖

## 三、解决方案

### 方案1：修复显存问题（推荐）

#### 方法A：先加载到CPU，再移动到GPU

修改`validation.py`第217行：

```python
# 原代码
model.load_state_dict(torch.load(args.weight_path, map_location = device))

# 修改为
model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
model.to(device)
```

**优点**：
- 避免加载时占用GPU显存
- 更灵活的内存管理

#### 方法B：清理GPU显存

在运行评估前，先清理GPU：

```bash
# 查看GPU使用情况
nvidia-smi

# 如果发现其他进程占用，可以：
# 1. 等待其他进程完成
# 2. 或使用其他GPU（修改脚本中的device参数）
```

#### 方法C：使用CPU加载，延迟移动到GPU

```python
# 修改validation.py
model = ClipCaptionModel(...)
# 先加载到CPU
model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
# 加载CLIP和其他组件后再移动到GPU
model.to(device)
```

#### 方法D：设置PyTorch内存分配策略

根据错误提示，可以设置环境变量：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/eval_coco.sh train_coco 0 '' 14
```

### 方案2：修复依赖问题

#### 安装scikit-image

```bash
# 激活环境
conda activate viecap

# 安装scikit-image
pip install scikit-image

# 或者使用conda
conda install scikit-image
```

#### 检查其他可能缺失的依赖

```bash
# 检查pycocotools相关依赖
pip install pycocotools
pip install scikit-image
pip install matplotlib
```

## 四、完整修复步骤

### 步骤1：修改代码（已自动修复）

**已修复的文件**：
- `validation.py` - 第217行
- `infer_by_instance.py` - 第57行  
- `infer_by_batch.py` - 第60行

**修改内容**：
```python
# 原代码
model.load_state_dict(torch.load(args.weight_path, map_location = device))

# 修改为
model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
```

所有相关文件已自动修复，无需手动修改。

### 步骤2：安装缺失依赖

```bash
conda activate viecap
pip install scikit-image
```

### 步骤3：清理GPU显存（可选）

```bash
# 查看GPU使用
nvidia-smi

# 如果有其他进程占用，考虑：
# - 等待进程完成
# - 使用其他GPU（修改device参数）
# - 或重启相关进程
```

### 步骤4：重新运行评估

```bash
bash scripts/eval_coco.sh train_coco 0 '' 14
```

## 五、代码修改示例

### 修改validation.py

```python
# validation.py 第214-218行
# loading model
tokenizer = AutoTokenizer.from_pretrained(args.language_model)
model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, gpt_type = args.language_model)

# 修改前：
# model.load_state_dict(torch.load(args.weight_path, map_location = device))
# model.to(device)

# 修改后：
model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
model.to(device)
```

**完整修改后的代码段**：

```python
# loading model
tokenizer = AutoTokenizer.from_pretrained(args.language_model)
model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, gpt_type = args.language_model)

# 先加载到CPU，避免显存不足
model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
model.to(device)

if not args.using_image_features:
    encoder, preprocess = clip.load(args.clip_model, device = device)
    inpath = args.path_of_val_datasets
else:
    inpath = args.path_of_val_datasets[:-5] + f'_{clip_name}.pickle'
```

## 六、其他优化建议

### 6.1 使用预提取图像特征

评估时使用预提取的图像特征可以节省显存：

```bash
# 确保使用 --using_image_features 参数
# 在eval_coco.sh中已经包含此参数
```

### 6.2 批量大小优化

如果仍然显存不足，可以考虑：
- 减小评估时的batch size（如果代码支持）
- 使用梯度检查点（gradient checkpointing）

### 6.3 使用其他GPU

如果GPU 0被占用，可以使用其他GPU：

```bash
# 修改脚本中的device参数
bash scripts/eval_coco.sh train_coco 2 '' 14  # 使用GPU 2
```

## 七、验证修复

修复后，重新运行评估应该能够：

1. ✅ 成功加载模型权重
2. ✅ 正常运行评估流程
3. ✅ 生成评估结果JSON文件
4. ✅ 成功运行COCO评估脚本

## 八、常见问题

### Q1: 修改后仍然显存不足？

**解决方案**：
1. 检查是否有其他Python进程占用GPU
2. 尝试使用其他GPU
3. 确保使用预提取的图像特征（`--using_image_features`）

### Q2: 安装scikit-image后仍有问题？

**检查**：
```bash
# 验证安装
python -c "import skimage; print(skimage.__version__)"

# 如果仍有问题，尝试重新安装
pip uninstall scikit-image
pip install scikit-image
```

### Q3: 评估速度慢？

**优化建议**：
1. 使用预提取图像特征
2. 使用GPU加速
3. 减小beam search宽度（如果允许）

## 九、预期结果

修复后，评估应该能够：

1. **成功加载模型**：无显存错误
2. **处理所有图像**：生成描述
3. **保存结果**：`checkpoints/train_coco/coco*.json`
4. **计算指标**：BLEU, METEOR, CIDEr, SPICE

**典型输出**：
```
Loading model...
Processing images: 100%|██████████| 5000/5000 [XX:XX<00:00, X.XXit/s]
Saving results to checkpoints/train_coco/coco_generated_captions.json
Evaluating...
BLEU@4: 27.2
METEOR: 24.8
CIDEr: 92.9
SPICE: 18.2
```

## 十、总结

**主要问题**：
1. GPU显存被占用，加载模型时直接映射到GPU导致OOM
2. 缺少scikit-image依赖

**解决方案**：
1. 修改模型加载方式：先加载到CPU，再移动到GPU
2. 安装scikit-image：`pip install scikit-image`

**修改位置**：
- `validation.py` 第217行

**依赖安装**：
```bash
pip install scikit-image
```

---

**注意**：如果修改代码后仍有问题，请检查：
1. GPU显存是否真的被释放
2. 所有依赖是否已正确安装
3. 模型权重文件是否存在且完整

