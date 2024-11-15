#!/bin/bash
#SBATCH --job-name=deepmd_zhh
#SBATCH --nodes=1 
#SBATCH --ntasks=56   # the node in CPUTot=56
#SBATCH --time=1-02:00:00
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH --mem=0
#SBATCH --partition=operation
#SBATCH --account=material_1


# 初始化 conda 并激活 conda 环境 
source ~/.bashrc
conda activate deepmd2.2.11-cpu

# 堆栈大小设置为无限制
ulimit -s unlimited

# get the current time
echo "Job start time: $(date)"

# ------- train -------
# srun --partition=operation dp train input.json
# srun --partition=operation dp freeze
# srun --partition=operation dp compress      # frozen_model_compressed.pb
# ------- train end -------

# get the input from the command line

# todo  srun mpirun -np 1 python run.py   this is not working       
srun python run.py $1 $2 $3 %J_energy.txt

# get the current time
echo "Job end time: $(date)"
