#!/bin/bash

cd "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/"

# 设置变量
SOURCE_DIR="/mnt/spaceai-internal/panda-intervid/disk1/internvid"  # 原始 tar 文件存放目录
TEMP_DIR="/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/temp_extract"  # 临时解压目录
BOS_TARGET_DIR="bos://spaceai-internal/panda-intervid/internvid"  # 目标 BOS 存储路径
TOKEN="58477ba3d08178bca9faf332867db5fc"  # 认证 Token
LOG_DIR="/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/logs"  # 日志文件存放目录
UPLOADED_FILES_LOG="$LOG_DIR/uploaded_files.log"  # 记录已上传文件的日志

# 创建临时目录和日志文件
mkdir -p "$LOG_DIR"
touch "$UPLOADED_FILES_LOG"

# 使用 glob 语法扩展文件列表
files=(${SOURCE_DIR}/*.tar)  # 文件列表

echo "找到的 tar 文件: ${files[@]}"  # 输出找到的文件列表

# 检查是否有 .tar 文件
if [ ${#files[@]} -eq 0 ]; then
    echo "未找到 .tar 文件，退出"
    exit 1
fi

# 遍历 .tar 文件并处理
for file in "${files[@]}"; do
    mkdir -p "$TEMP_DIR"
    # 检查文件是否已上传
    if grep -Fxq "$file" "$UPLOADED_FILES_LOG"; then
        echo "文件 $file 已上传，跳过"
        continue
    fi

    echo "解压文件: $file 到 $TEMP_DIR"
    tar --checkpoint=100000 --checkpoint-action=echo="解压进度: %u 个文件" -xf "$file" --skip-old-files -C "$TEMP_DIR"
    echo "解压完成: $file"

    echo "上传解压后的文件到 BOS"
    adt push --no-ark-prefix --recursive --concurrency 2 \
        --src-dir "$TEMP_DIR" --target-dir "$BOS_TARGET_DIR" \
        --token "$TOKEN" \
        | grep -v "successfully pushed to:"

    # 记录已上传的文件
    echo "$file" >> "$UPLOADED_FILES_LOG"
    echo "文件 $file 上传完成，已记录"

    echo "清理临时目录 $TEMP_DIR"
    rm -rf "$TEMP_DIR"
done

# 同步 BOS 数据到本地挂载点
echo "同步 BOS 数据到本地"
adt syncmeta --follow --path "/panda-intervid/internvid" --token "$TOKEN"

echo "任务完成！"
