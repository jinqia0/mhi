#!/bin/bash
# ========== 系统配置 ==========
# 文件名：run_pllava_cluster.sh
# 功能：PLLaVA多卡视频描述生成启动脚本
# 版本：2.0 (适配A800集群)

# ========== 硬件配置 ==========
export CUDA_VISIBLE_DEVICES=0,1                # 使用前两块A800 GPU
export NCCL_DEBUG=INFO                         # 开启NCCL调试信息
export NCCL_IB_DISABLE=0                       # 启用InfiniBand
export NCCL_SOCKET_IFNAME=eth0                 # 指定网络接口
export FI_PROVIDER=efa                         # 使用AWS弹性适配器
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,expandable_segments:True" # 内存优化

# ========== 日志配置 ==========
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="cluster_logs/${TIMESTAMP}"
mkdir -p ${LOG_DIR}
exec > >(tee -a "${LOG_DIR}/stdout.log") 2> >(tee -a "${LOG_DIR}/stderr.log" >&2)

# ========== 模型参数 ==========
PRETRAINED_MODEL="/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/PLLaVA/MODELS/pllava-13b"
CSV_PATH="../data/internvid/internvid_00_100.csv"  # 输入数据路径
NUM_FRAMES=16                                      # 视频采样帧数
BATCH_SIZE=32                                      # 单卡批大小
GRAD_ACCUM_STEPS=2                                 # 梯度累积步数

# ========== 加速器配置 ==========
ACCELERATE_CONFIG="configs/a800_accelerate.yaml"   # 加速器配置文件
MASTER_PORT=29501                                  # 主节点端口

# ========== 执行命令 ==========
echo "[$(date)] 开始运行PLLaVA多卡生成任务" | tee -a ${LOG_DIR}/metrics.log

/mnt/pfs-gv8sxa/tts/dhg/jinqiao/envs/pllava/bin/accelerate launch \
--config_file $ACCELERATE_CONFIG \
/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/PLLaVA/caption_pllava.py \
--pretrained_model_name_or_path $PRETRAINED_MODEL \
--csv_path $CSV_PATH \
--num_frames $NUM_FRAMES \
--batch_size $BATCH_SIZE \
--pooling_shape "16-12-12" \
--mixed_precision bf16 \
--use_multi_gpus \
--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
--max_gpu_mem_util 0.95 \                         # 最大显存利用率
--enable_flash_attn \                             # 启用Flash Attention
--video_resolution 672 \                          # 视频分辨率
--log_dir $LOG_DIR

# ========== 运行后检查 ==========
echo "[$(date)] 任务完成，退出码: $?" | tee -a ${LOG_DIR}/metrics.log

# 收集显存使用统计
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free \
--format=csv -l 1 > ${LOG_DIR}/gpu_stats.csv &

# 生成性能报告
python tools/analyze_logs.py --log-dir $LOG_DIR --output ${LOG_DIR}/performance_report.html
