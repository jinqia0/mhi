#!/bin/bash

# CSV文件分割脚本
# 将超大CSV文件分割成<2GB的小文件，适配GitHub LFS限制

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔪 MHI项目大文件分割工具${NC}"
echo "============================================="

# 分割函数
split_csv_file() {
    local input_file="$1"
    local lines_per_split="$2"
    local output_prefix="$3"

    if [[ ! -f "$input_file" ]]; then
        echo -e "${RED}❌ 文件不存在: $input_file${NC}"
        return 1
    fi

    echo -e "${YELLOW}📊 正在分析文件: $input_file${NC}"
    local total_lines=$(wc -l < "$input_file")
    local file_size=$(du -h "$input_file" | cut -f1)
    echo "   总行数: $total_lines"
    echo "   文件大小: $file_size"

    # 创建输出目录
    local output_dir=$(dirname "$output_prefix")
    mkdir -p "$output_dir"

    echo -e "${YELLOW}✂️  开始分割 (每片 $lines_per_split 行)...${NC}"

    # 保存原始头部
    local header=$(head -1 "$input_file")
    echo "   CSV头部: $header"

    # 计算预期分片数
    local expected_parts=$(( (total_lines + lines_per_split - 1) / lines_per_split ))
    echo "   预期分片数: $expected_parts"

    # 分割文件 (跳过头部)
    tail -n +2 "$input_file" | split -l "$lines_per_split" - "${output_prefix}"

    # 为每个分片添加头部
    local part_count=0
    for part_file in "${output_prefix}"*; do
        if [[ -f "$part_file" ]]; then
            # 创建临时文件
            local temp_file="${part_file}.tmp"
            echo "$header" > "$temp_file"
            cat "$part_file" >> "$temp_file"
            mv "$temp_file" "${part_file}.csv"
            rm "$part_file"

            part_count=$((part_count + 1))
            local part_size=$(du -h "${part_file}.csv" | cut -f1)
            echo "   ✅ 创建分片 $part_count: ${part_file}.csv ($part_size)"
        fi
    done

    echo -e "${GREEN}✅ 分割完成! 共创建 $part_count 个分片${NC}"
}

# 分割配置
declare -A SPLIT_CONFIG=(
    # 文件路径:每片行数:输出前缀
    ["csv/data/panda/panda.csv"]="1900000:csv/data/panda/parts/panda_part_"
    ["csv/data/internvid/internvid_yolo.csv"]="1000000:csv/data/internvid/parts/internvid_yolo_part_"
    ["csv/data/internvid/internvid.csv"]="800000:csv/data/internvid/parts/internvid_part_"
    ["csv/data/panda/panda_multi_aes45.csv"]="600000:csv/data/panda/parts/panda_multi_aes45_part_"
    ["csv/data/mhi/mhi_multi_aes45_coarse_ocr.csv"]="500000:csv/data/mhi/parts/mhi_multi_aes45_coarse_ocr_part_"
)

# 执行分割
for file_path in "${!SPLIT_CONFIG[@]}"; do
    if [[ -f "$file_path" ]]; then
        config="${SPLIT_CONFIG[$file_path]}"
        lines_per_split="${config%:*}"
        output_prefix="${config#*:}"

        echo ""
        split_csv_file "$file_path" "$lines_per_split" "$output_prefix"
    else
        echo -e "${YELLOW}⚠️  跳过不存在的文件: $file_path${NC}"
    fi
done

echo ""
echo -e "${GREEN}🎉 所有文件分割完成!${NC}"
echo "============================================="
echo "📁 分片文件位置:"
find csv/data -name "*_part_*.csv" -type f | head -10
echo ""
echo "💾 磁盘使用情况:"
du -sh csv/data/*/parts/ 2>/dev/null || echo "   (暂无分片目录)"