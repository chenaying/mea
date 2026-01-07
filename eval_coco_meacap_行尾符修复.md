# eval_coco_meacap.sh 行尾符修复说明

## 一、问题描述

在 Linux 系统上运行脚本时出现错误：
```
scripts/eval_coco_meacap.sh: 行 3: $'\r'：未找到命令
```

**原因**：脚本文件使用了 Windows 行尾符（CRLF `\r\n`），而 Linux 系统期望 Unix 行尾符（LF `\n`）。

## 二、解决方案

### 方案 1：在 Linux 上使用 dos2unix 工具（推荐）

```bash
# 安装 dos2unix（如果没有）
sudo apt-get install dos2unix  # Ubuntu/Debian
# 或
sudo yum install dos2unix      # CentOS/RHEL

# 转换文件
dos2unix scripts/eval_coco_meacap.sh

# 设置执行权限
chmod +x scripts/eval_coco_meacap.sh
```

### 方案 2：使用 sed 命令转换

```bash
# 移除所有 \r 字符
sed -i 's/\r$//' scripts/eval_coco_meacap.sh

# 设置执行权限
chmod +x scripts/eval_coco_meacap.sh
```

### 方案 3：使用 tr 命令转换

```bash
# 移除所有 \r 字符
tr -d '\r' < scripts/eval_coco_meacap.sh > scripts/eval_coco_meacap.sh.tmp
mv scripts/eval_coco_meacap.sh.tmp scripts/eval_coco_meacap.sh

# 设置执行权限
chmod +x scripts/eval_coco_meacap.sh
```

### 方案 4：使用 Git 自动转换（预防）

在 Windows 上配置 Git 自动处理行尾符：

```bash
# 配置 Git 在提交时自动转换 CRLF 为 LF
git config core.autocrlf true

# 对于 shell 脚本，强制使用 LF
git config core.eol lf

# 或者为 .sh 文件设置特定属性
echo "*.sh text eol=lf" >> .gitattributes
```

## 三、验证修复

修复后，验证脚本是否正常：

```bash
# 检查行尾符（应该只显示 $，不显示 \r）
cat -A scripts/eval_coco_meacap.sh | head -5

# 应该看到行尾只有 $，没有 ^M
# 如果看到 ^M$，说明还有 CRLF
```

## 四、预防措施

### 4.1 创建 .gitattributes 文件

在项目根目录创建 `.gitattributes` 文件：

```
# Shell scripts always use LF
*.sh text eol=lf

# Python files use LF
*.py text eol=lf

# Markdown files use LF
*.md text eol=lf
```

### 4.2 配置编辑器

在 VS Code/Cursor 中：
1. 打开设置
2. 搜索 "eol"
3. 设置 `Files: Eol` 为 `\n`（LF）

### 4.3 使用 Git 钩子（可选）

在 `.git/hooks/pre-commit` 中添加检查：

```bash
#!/bin/bash
# 检查 shell 脚本是否有 CRLF
for file in $(git diff --cached --name-only --diff-filter=ACM | grep '\.sh$'); do
    if file "$file" | grep -q CRLF; then
        echo "Error: $file has CRLF line endings"
        exit 1
    fi
done
```

## 五、重新运行脚本

修复后，重新运行：

```bash
bash scripts/eval_coco_meacap.sh train_coco 0 '' 14
```

## 六、总结

**问题**：Windows 行尾符（CRLF）在 Linux 上导致脚本无法执行

**解决方法**：
1. ✅ 使用 `dos2unix` 转换文件（最简单）
2. ✅ 使用 `sed` 或 `tr` 移除 `\r` 字符
3. ✅ 配置 Git 自动转换（预防未来问题）

**预防**：
- 创建 `.gitattributes` 文件
- 配置编辑器使用 LF
- 使用 Git 钩子检查

