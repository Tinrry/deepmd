#!/bin/bash
#SBATCH -J deepmd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p debug

srun python sys_atoms.py
