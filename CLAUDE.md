# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Multi-modal Human Interaction (MHI) repository - a comprehensive video processing pipeline for analyzing human interactions in video datasets. The project focuses on analyzing human contact detection, depth estimation, object detection, and aesthetic scoring across large-scale video datasets.

## Key Commands

### Claude Code快捷命令
```bash
# 使用askc别名快速启动Claude Code
askc  # 等价于 claude -p
```

### Environment Setup
```bash
# Activate conda environment
conda activate mhi

# Install dependencies
pip install -r requirements.txt

# Download required model checkpoints
bash download_models.sh
```

### Core Processing Commands

#### YOLO-based Human Detection
```bash
# Single GPU processing
python filters/yolo/yolo_single.py <gpu_id> <csv_path>

# Multi-GPU parallel processing
bash filters/yolo/mp_bash.sh

# Multi-process with memory optimization
python filters/yolo/yolo_mp.py
```

#### Depth-Enhanced Contact Analysis
```bash
# Run parallel depth contact analysis
python core/yolo_depth_parallel.py

# Run basic depth contact analysis
python core/test_depth_contact.py
```

#### Video Annotation Interface
```bash
# Launch Streamlit annotation interface
streamlit run core/video_annotator.py --server.port 8501 --server.headless true
```

#### Aesthetic Scoring
```bash
# Process videos for aesthetic scores
python filters/aesthetic/aes.py

# Multi-frame aesthetic processing
python filters/aesthetic/aes_mf.py
```

#### Data Extraction and Processing
```bash
# Extract video clips from compressed archives
bash extract_clips.sh

# Process CSV files with utilities
python utils/process_csv_videos.py
python utils/process_openhv_filtered.py

# Depth video processing
python depth_video_processor.py

# Batch video pose extraction
python batch_video_pose_extraction.py --input-dir videos --output-dir pose_results
```

### Testing Commands
```bash
# Test depth contact detection
python core/test_depth_contact.py

# Run mask analysis
python core/run_mask_analysis.py

# Run depth analysis
python core/run_depth_analysis.py
```

### Analysis and Evaluation Commands
```bash
# Analyze contact detection accuracy against manual annotations
python analysis_contact_detection_accuracy.py

# Progressive enhancement analysis
python progressive_enhancement_analysis.py

# Extract misclassified videos for review
python extract_misclassified_videos.py

# Extract enhancement samples for further analysis
python extract_enhancement_samples.py
```

## Architecture Overview

### Core Processing Pipeline
The repository implements a multi-stage video analysis pipeline:

1. **Video Detection Layer** (`filters/`): YOLO-based human detection, aesthetic scoring, OCR processing
2. **Core Analysis Layer** (`core/`): Depth estimation, contact detection, mask analysis  
3. **Utilities Layer** (`utils/`): Parallel processing frameworks, CSV handling, data utilities
4. **Data Management** (`Dataset/`): Large-scale video datasets (OpenHumanVid, Panda)

### Key Components

#### Parallel Processing Framework (`utils/parallel_processor.py`)
- `GPUWorkerManager`: Manages multi-GPU worker processes
- `BatchProcessor`: Handles batch processing with configurable sizes
- `ChunkedDataProcessor`: Processes large CSV files in chunks
- `BaseGPUWorker`: Abstract base class for GPU-accelerated workers

#### Video Analysis Core (`core/`)
- `yolo_depth_parallel.py`: Combines YOLO detection with depth estimation for contact analysis
- `video_annotator.py`: Streamlit-based manual annotation interface
- `mask_analysis_worker.py`: Analyzes mask changes for motion detection
- `yolo_seg.py`: YOLO segmentation with depth integration

#### Filter Modules (`filters/`)
- `yolo/`: Human detection with IoU-based interaction analysis
- `aesthetic/`: CLIP-based aesthetic scoring using pre-trained models  
- `coarse/`: Coarse-grained filtering for video quality
- `ocr/`: Text detection and recognition in videos
- `llama/`: LLaMA-based language model processing for video understanding

### Data Flow Architecture

1. **Input**: Large CSV files containing video paths and metadata
2. **Chunked Processing**: Data split into manageable chunks for parallel processing
3. **GPU Distribution**: Tasks distributed across multiple GPUs using worker pools
4. **Batch Inference**: Models process multiple videos simultaneously for efficiency
5. **Results Aggregation**: Processed chunks merged into final output CSV files

### Configuration Patterns

#### GPU Optimization Settings
```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

#### Common Processing Parameters
- `BATCH_SIZE`: Typically 8-32 depending on GPU memory
- `CONF_THRESH`: 0.6 for YOLO detection confidence
- `IOU_THRESH`: 0.45 for non-maximum suppression
- `POOL_SIZE_PER_GPU`: 2-4 processes per GPU

### Dataset Structure

#### OpenHumanVid Dataset
- Location: `Dataset/OpenHumanVid/`
- Format: Compressed `.tgz` archives with corresponding CSV metadata
- Processing: Use `extract_clips.sh` for extraction

#### Panda Dataset  
- Location: `Dataset/panda/`
- Format: Direct MP4 files organized by subdirectories
- Metadata: CSV files with video paths and annotations

### Model Checkpoints
- YOLO models: `checkpoints/yolo11*.pt`
- Depth estimation: `checkpoints/depth_anything_v2_vits.pth`
- Pose estimation: `checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth`
- Human detection: `checkpoints/faster_rcnn_r50_fpn_1x_coco-person_*.pth`
- Action recognition: `checkpoints/stgcn_8xb16-*.pth` (ST-GCN models for skeleton-based action recognition)
- Aesthetic scoring: External CLIP-based models

### Results and Output
- Processing results stored in `results/` directory
- CSV format with columns for detection metrics, scores, and analysis results
- Annotation results from manual review stored separately

### Performance Considerations

#### Memory Management
- Use chunked processing for large datasets
- Implement batch processing to maximize GPU utilization
- Clear CUDA cache between processing chunks

#### Parallel Processing
- Distribute work across multiple GPUs using process pools
- Use separate processes per GPU to avoid GIL limitations
- Implement proper error handling and recovery mechanisms

### Integration with Third-Party Tools
- `third_party/Depth-Anything-V2/`: Depth estimation models and utilities
- `third_party/Open-Sora/`: Video generation and processing utilities (removed but may be referenced)
- `third_party/PLLaVA/`: Multi-modal language model integration (removed but may be referenced)

### Environment Requirements
```bash
# Python environment with conda
conda activate mhi

# Key dependencies (see requirements.txt for full list)
torch==2.6.0+cu118
torchvision==0.21.0+cu118
ultralytics==8.3.98
opencv-python==4.11.0.86
transformers==4.49.0
```

## Development Workflow

1. **Data Preparation**: Extract and organize video datasets
2. **Filter Processing**: Run detection and scoring filters 
3. **Core Analysis**: Apply depth and interaction analysis
4. **Manual Review**: Use annotation interface for quality control
5. **Results Analysis**: Process and analyze final CSV outputs

### Key Processing Patterns

#### Torch Environment Setup
Many scripts include this pattern for PyTorch logging fixes:
```python
import os
os.environ.update({
    'PYTHONWARNINGS': 'ignore',
    'TOKENIZERS_PARALLELISM': 'false',
})
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

#### Multiprocessing Usage
Use `mp.set_start_method('spawn', force=True)` for CUDA compatibility in multiprocessing contexts.

#### CSV Processing Workflow
The typical processing workflow involves:
1. Load CSV files with video metadata from `Dataset/` directories
2. Process videos using parallel GPU workers
3. Output results to `results/` directory with enhanced metadata
4. Batch processing with configurable chunk sizes for memory management

#### Log Management
- Processing logs stored in `logs/` directory organized by filter type
- Each GPU process writes to separate log files for parallel debugging
- Log rotation and cleanup handled automatically

When working with this codebase, always consider GPU memory constraints and use the parallel processing framework for scalability across large datasets.

## User Instructions
- 用中文回答
- 对于复杂的任务，先列出预期实现的功能与todo，确保理解没有偏差