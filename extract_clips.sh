#!/bin/bash

# 解压clips文件夹中的tgz压缩包脚本
# 边解压边删除原压缩包以节省空间
# 支持并行处理加速解压

set -e  # 遇到错误时退出

# 并行处理配置
MAX_PARALLEL_JOBS=4  # 最大并行任务数，可根据CPU核心数调整

# 设置路径
CLIPS_DIR="Dataset/OpenHumanVid/clips"
LOG_FILE="$(pwd)/logs/extract_clips.log"

# 日志函数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# 检查目录是否存在
if [ ! -d "$CLIPS_DIR" ]; then
    log "错误: 目录不存在: $CLIPS_DIR"
    exit 1
fi

# 进入clips目录
cd "$CLIPS_DIR"

# 获取所有tgz文件并排序
TGZ_FILES=($(ls -1 *.tgz 2>/dev/null | sort -V))

if [ ${#TGZ_FILES[@]} -eq 0 ]; then
    log "没有找到tgz文件"
    exit 0
fi

log "找到 ${#TGZ_FILES[@]} 个tgz文件"

# 计算总大小
TOTAL_SIZE=0
for file in "${TGZ_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(stat -c%s "$file")
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    fi
done

TOTAL_SIZE_GB=$((TOTAL_SIZE / 1024 / 1024 / 1024))
log "总大小: ${TOTAL_SIZE_GB} GB"

# 确认操作
echo "确认要解压 ${#TGZ_FILES[@]} 个文件并删除原压缩包吗？(y/N): "
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    log "操作已取消"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAILED_COUNT=0

# 并行解压函数
extract_file() {
    local file="$1"
    local file_num="$2"
    local total_num="$3"
    
    log "开始处理第 ${file_num}/${total_num} 个文件: $file (PID: $$)"
    
    # 获取文件大小
    if [ -f "$file" ]; then
        SIZE=$(stat -c%s "$file")
        SIZE_MB=$((SIZE / 1024 / 1024))
        log "文件大小: ${SIZE_MB} MB (文件: $file)"
    fi
    
    # 使用pigz进行多线程解压（如果可用），否则使用普通tar
    if command -v pigz >/dev/null 2>&1; then
        # 使用pigz进行多线程解压
        if tar -I pigz -xf "$file"; then
            rm "$file"
            log "解压成功并删除原文件: $file (使用pigz多线程)"
            return 0
        else
            log "使用pigz解压失败: $file"
            return 1
        fi
    else
        # 普通tar解压，但使用更高效的参数
        if tar -xzf "$file" --checkpoint=1000 --checkpoint-action=dot; then
            rm "$file"
            log "解压成功并删除原文件: $file (使用tar)"
            return 0
        else
            log "解压失败: $file"
            return 1
        fi
    fi
}

# 导出函数和变量供子进程使用
export -f extract_file log
export LOG_FILE

# 使用GNU parallel或xargs进行并行处理
if command -v parallel >/dev/null 2>&1; then
    log "使用GNU parallel进行并行解压 (${MAX_PARALLEL_JOBS} 个并行任务)"
    
    # 创建任务列表
    for i in "${!TGZ_FILES[@]}"; do
        echo "${TGZ_FILES[$i]} $((i + 1)) ${#TGZ_FILES[@]}"
    done | parallel -j "$MAX_PARALLEL_JOBS" --colsep ' ' extract_file {1} {2} {3}
    
    # 统计结果
    SUCCESS_COUNT=$(ls -1 2>/dev/null | wc -l || echo 0)
    FAILED_COUNT=$((${#TGZ_FILES[@]} - SUCCESS_COUNT))
    
else
    log "使用批量并行解压 (${MAX_PARALLEL_JOBS} 个并行任务)"
    
    # 手动实现并行处理
    job_count=0
    for i in "${!TGZ_FILES[@]}"; do
        file="${TGZ_FILES[$i]}"
        current_num=$((i + 1))
        total_num=${#TGZ_FILES[@]}
        
        # 启动后台任务
        extract_file "$file" "$current_num" "$total_num" &
        
        # 控制并发数量
        job_count=$((job_count + 1))
        if [ $job_count -ge $MAX_PARALLEL_JOBS ]; then
            wait  # 等待所有后台任务完成
            job_count=0
            
            # 计算进度
            ELAPSED_TIME=$(($(date +%s) - START_TIME))
            REMAINING_FILES=$((total_num - current_num))
            if [ $current_num -gt 0 ] && [ $ELAPSED_TIME -gt 0 ]; then
                AVG_TIME_PER_FILE=$((ELAPSED_TIME / current_num))
                ESTIMATED_REMAINING_TIME=$((REMAINING_FILES * AVG_TIME_PER_FILE))
                ESTIMATED_HOURS=$((ESTIMATED_REMAINING_TIME / 3600))
                log "已完成批次，预计剩余时间: ${ESTIMATED_HOURS} 小时"
            fi
        fi
    done
    
    # 等待最后一批任务完成
    wait
    
    # 统计结果 - 通过检查原始tgz文件是否还存在来判断成功/失败
    SUCCESS_COUNT=0
    FAILED_COUNT=0
    for file in "${TGZ_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done
fi

# 完成统计
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_TIME / 3600))

log "解压完成！"
log "总耗时: ${TOTAL_HOURS} 小时"
log "成功: ${SUCCESS_COUNT}, 失败: ${FAILED_COUNT}"

# 显示磁盘使用情况
log "当前目录磁盘使用情况:"
du -sh . 2>/dev/null || log "无法获取磁盘使用情况" 