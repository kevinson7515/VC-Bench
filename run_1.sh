#!/bin/bash
#SBATCH --job-name=wan_video # 指定作业名
#SBATCH -o /home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/wan_result # 指定输出文件名;默认为slurm-[jobID].out
#SBATCH -e /home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/wan_result  # 指定程序错误输出文件名
#SBATCH --partition=q_intel_gpu_nvidia_h20 # 指定作业提交的分区
#SBATCH --gres=gpu:2 # 指定作业使用卡的总数为 4


# 加载CUDA模块
# module load amd/cuda/11.8.89
# module load intel/gcc_compiler/10.3.0

# 初始化conda环境
source ~/.bashrc
conda activate VBench

# 如果你的项目需要代理，可以取消注释下面这两行
# export http_proxy="http://174.0.250.13:3128"
# export https_proxy="http://174.0.250.13:3128"

# 切换到脚本所在目录
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset http_proxy
unset https_proxy

# 根据实际路径修改
# cd /home/export/base/ycsc_chenkh/hitici_08/online1/Files/phenaki-pytorch-main/phenaki_pytorch
python -m torch.distributed.run --nproc_per_node=2 --master_port 29500 /home/export/base/ycsc_chenkh/hitici_08/online1/Files/VC_benchmark/benchmark/connect_distance.py