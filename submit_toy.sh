#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH -t 0-01:00
#SBATCH --mem=16G
#SBATCH -o slurm_logs/job_%j.out
#SBATCH -e slurm_logs/job_%j.err

# load modules
source ~/.bash_profile
mamba activate torch_gpu

cd /n/home11/sambt/contrastive_anomaly/training_JetClass/NPLM-embedding

dim=$1
temp=$2
inputs=$3
nref=$4
nbkg=$5
nsig=$6

python toy.py --dim $dim --temp $temp --inputs $inputs -r $nref -b $nbkg -s $nsig -t 100 -l 4 --fractions