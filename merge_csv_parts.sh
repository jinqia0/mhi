#!/bin/bash

# CSV文件合并脚本
# 将分片的CSV文件重新合并为完整文件

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔗 MHI项目文件合并工具${NC}"
echo "============================================="

# 合并函数
merge_csv_parts() {
    local parts_pattern="$1"
    local output_file="$2"
    local description="$3"

    echo -e "${YELLOW}📋 合并: $description${NC}"
    echo "   模式: $parts_pattern"
    echo "   输出: $output_file"

    # 检查分片文件是否存在
    local part_files=($(ls $parts_pattern 2>/dev/null | sort))
    if [[ ${#part_files[@]} -eq 0 ]]; then
        echo -e "${RED}❌ 未找到分片文件: $parts_pattern${NC}"
        return 1
    fi

    echo "   找到 ${#part_files[@]} 个分片文件"

    # 创建输出目录
    local output_dir=$(dirname "$output_file")
    mkdir -p "$output_dir"

    # 从第一个文件获取头部
    local header=$(head -1 "${part_files[0]}")
    echo "   CSV头部: $header"

    # 写入头部到输出文件
    echo "$header" > "$output_file"

    # 合并所有分片（跳过每个分片的头部）
    local total_lines=0
    for part_file in "${part_files[@]}"; do
        local part_lines=$(tail -n +2 "$part_file" | wc -l)
        tail -n +2 "$part_file" >> "$output_file"
        total_lines=$((total_lines + part_lines))
        echo "   ✅ 合并分片: $(basename "$part_file") (${part_lines} 行)"
    done

    # 验证合并结果
    local final_lines=$((total_lines + 1))  # +1 for header
    local actual_lines=$(wc -l < "$output_file")
    local file_size=$(du -h "$output_file" | cut -f1)

    if [[ $final_lines -eq $actual_lines ]]; then
        echo -e "${GREEN}✅ 合并成功! 总行数: $actual_lines, 文件大小: $file_size${NC}"
    else
        echo -e "${RED}❌ 行数不匹配! 预期: $final_lines, 实际: $actual_lines${NC}"
        return 1
    fi
}

# 合并配置
declare -A MERGE_CONFIG=(
    # 分片模式:输出文件:描述
    ["csv/data/panda/parts/panda_part_*.csv"]="csv/data/panda/panda_merged.csv:Panda主数据集"
    ["csv/data/internvid/parts/internvid_yolo_part_*.csv"]="csv/data/internvid/internvid_yolo_merged.csv:InternVid YOLO数据"
    ["csv/data/internvid/parts/internvid_part_*.csv"]="csv/data/internvid/internvid_merged.csv:InternVid原始数据"
    ["csv/data/panda/parts/panda_multi_aes45_part_*.csv"]="csv/data/panda/panda_multi_aes45_merged.csv:Panda多维分析数据"
    ["csv/data/mhi/parts/mhi_multi_aes45_coarse_ocr_part_*.csv"]="csv/data/mhi/mhi_multi_aes45_coarse_ocr_merged.csv:MHI综合分析数据"
)

# 执行合并
echo -e "${BLUE}🎯 开始合并分片文件...${NC}"
echo ""

for pattern in "${!MERGE_CONFIG[@]}"; do
    config="${MERGE_CONFIG[$pattern]}"
    output_file="${config%:*}"
    description="${config#*:}"

    # 检查是否存在分片文件
    if ls $pattern 1> /dev/null 2>&1; then
        merge_csv_parts "$pattern" "$output_file" "$description"
        echo ""
    else
        echo -e "${YELLOW}⚠️  跳过不存在的分片: $pattern${NC}"
        echo ""
    fi
done

echo -e "${GREEN}🎉 所有文件合并完成!${NC}"
echo "============================================="
echo -e "${BLUE}📊 合并后文件统计:${NC}"
find csv/data -name "*_merged.csv" -exec ls -lh {} \; | while read line; do
    echo "   $line"
done

echo ""
echo -e "${BLUE}💡 使用说明:${NC}"
echo "   - 原始大文件已保留，可以删除节省空间"
echo "   - 分片文件位于 parts/ 子目录中"
echo "   - 合并文件添加了 '_merged' 后缀"
echo "   - 所有文件都保持原始CSV格式和结构"