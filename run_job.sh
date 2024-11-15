#!/bin/bash
#SBATCH --job-name=deepmd_zhh
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --time=1-02:00:00
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH --mem=0
#SBATCH --partition=operation

# 设置环境变量以优化并行性 
# export OMP_NUM_THREADS=4 
# export TF_INTRA_OP_PARALLELISM_THREADS=4 
# export TF_INTER_OP_PARALLELISM_THREADS=4

# 初始化 conda 并激活 conda 环境 
source ~/.bashrc
conda activate deepmd2.2.11-cpu

# 基础集群计算限制
ulimit -s unlimited

# ------- train -------
# srun --partition=operation dp train input.json
# srun --partition=operation dp freeze
# srun --partition=operation dp compress      # frozen_model_compressed.pb
# ------- train end -------

# todo  srun mpirun -np 1 python run.py   this is not working       
srun python run.py
