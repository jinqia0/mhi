#!/bin/bash

echo "启动视频标注系统..."
echo "========================================"

# 检查是否有Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 安装依赖（如果需要）
echo "检查依赖包..."
pip3 install flask pandas

# 启动服务器
echo "启动Flask服务器..."
echo "========================================"
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo "========================================"

python3 video_annotation_server.py