#!/bin/bash

echo "启动视频标注系统..."
echo "========================================"

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 安装依赖（如果需要）
echo "检查并安装依赖包..."
pip install flask requests

# 检查安装是否成功
if ! python -c "import flask" &> /dev/null; then
    echo "错误: Flask安装失败，请手动安装Flask"
    exit 1
fi

# 启动服务器
echo "启动Flask服务器..."
echo "========================================"
echo "访问地址: http://localhost:5001"
echo "按 Ctrl+C 停止服务器"
echo "========================================"

python video_annotation_server.py