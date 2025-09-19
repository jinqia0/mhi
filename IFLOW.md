# 项目概览

本项目是一个用于视频人物交互分析和标注的工具集，主要用于检测视频中人物之间的接触行为，并提供可视化标注界面。项目使用Python开发，集成了YOLO目标检测模型、Streamlit可视化框架等技术。

## 主要功能

1. **视频人物交互检测**：
   - 使用YOLO分割模型检测视频中的人物
   - 分析人物之间的直接接触（人与人接触）
   - 分析人物之间的间接接触（通过物体的接触）
   - 结合深度信息进行更精确的接触分析

2. **视频标注工具**：
   - 提供基于Streamlit的可视化标注界面
   - 支持多轮标注和多人协作
   - 支持键盘快捷键操作提高标注效率

3. **并行处理框架**：
   - 实现高效的视频并行处理机制
   - 支持多GPU加速处理
   - 可处理大规模视频数据集

## 技术栈

- Python 3.x
- YOLO (Ultralytics)
- Streamlit
- OpenCV
- PyTorch
- Pandas
- Numpy

# 项目结构

```
.
├── README.md
├── config.toml
├── requirements.txt
├── main.py
├── core/
│   ├── video_annotator.py
│   ├── yolo_seg.py
│   ├── yolo_seg_with_depth.py
│   ├── run_mask_analysis.py
│   ├── mask_analysis_worker.py
│   └── ...
├── filters/
│   ├── yolo/
│   ├── ocr/
│   └── ...
├── lizrun/
│   ├── yolo/
│   ├── ocr/
│   └── ...
└── third_party/
    └── Open-Sora/
```

# 核心组件

## 1. 视频标注工具 (core/video_annotator.py)

这是一个基于Streamlit的交互式视频标注工具，支持：
- 多数据源支持（OpenHumanVid、Panda等）
- 多轮标注（支持多个标注者）
- 键盘快捷键操作（1: 有交互, 0: 无交互, S: 跳过）
- 实时进度跟踪
- 标注结果预览和统计

## 2. 人物接触检测 (core/yolo_seg.py)

使用YOLO分割模型进行人物接触检测：
- 直接接触检测（人与人接触）
- 间接接触检测（通过物体的接触）
- 支持并行处理加速
- 输出详细的接触统计信息

## 3. Mask变化分析 (core/mask_analysis_worker.py, core/run_mask_analysis.py)

分析视频中人物mask的变化情况：
- 计算mask变化比例
- 识别主要人物
- 支持并行处理

## 4. 深度信息处理 (core/yolo_seg_with_depth.py)

结合深度信息进行更精确的接触检测：
- 深度图处理
- 3D点云生成
- 深度约束的接触分析

# 运行和构建

## 环境准备

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 下载YOLO模型权重文件到`checkpoints/`目录：
   - `yolo11n-seg.pt` (用于分割)
   - `yolo11n.pt` (用于检测)

## 运行视频标注工具

```bash
streamlit run core/video_annotator.py
```

## 运行接触检测

```bash
# 基础接触检测
python core/yolo_seg.py

# 结合深度信息的接触检测
python core/yolo_seg_with_depth.py

# Mask变化分析
python core/run_mask_analysis.py
```

## 并行处理参数

大部分处理脚本支持以下参数：
- `--frame-interval N`: 每N帧处理一次
- `--max-videos N`: 最多处理N个视频
- `--gpus ID1,ID2,...`: 指定使用的GPU ID
- `--chunk-size N`: 并行处理的块大小

# 开发约定

## 代码风格

- 遵循PEP 8代码规范
- 使用类型注解
- 函数和类需要适当的文档字符串

## 并行处理框架

项目实现了统一的并行处理框架：
- `BaseGPUWorker`: 基础GPU工作器类
- `ParallelVideoProcessor`: 并行视频处理器
- 支持多GPU和多进程并行处理

## 数据格式

CSV文件格式：
- 包含视频路径列
- 包含接触检测结果列
- 支持增量处理（只处理未处理的视频）