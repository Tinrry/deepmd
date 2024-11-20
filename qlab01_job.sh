#!/bin/bash
#SBATCH --job-name=deepmd_zhh
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --time=1-02:00:00
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH --mem=0
#SBATCH --partition=debug
#SBATCH --exclusive

# 初始化 conda 并激活 conda 环境 
source ~/.bashrc
source /sharedata01/hhzheng/deepmd2.2.10/bin/activate  /sharedata01/hhzheng/deepmd2.2.10

# 堆栈大小设置为无限制
ulimit -s unlimited

# ------- train -------
srun  dp train input_v2_compat.json
# srun --partition=operation dp freeze
# srun --partition=operation dp compress      # frozen_model_compressed.pb
# ------- train end -------

# get the input from the command line
# todo  error:srun mpirun -np 1 python run.py   this is not working       
# srun python run.py $1 $2 $3 ${SLURM_JOB_ID}_predict.txt
