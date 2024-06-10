#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBTACH --cpus-per-task=12
#SBATCH --mincpus=12
#SBATCH --job-name=SEEF00new
#SBATCH --output=log/output/gpu.%j.out
#SBATCH --error=log/error/gpu.%j.err
#SBATCH --gres=gpu:a40:1

#module load anaconda/3
#source activate base
python SEEF.py 0 0
