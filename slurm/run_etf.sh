#!/bin/bash

#SBATCH --job-name=etf_50_trt
#SBATCH --output=out/2026-03-30/%x.log
#SBATCH --error=out/2026-03-30/%x.log

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

#SBATCH --gres=gpu:L40S:1
#SBATCH --time=5-00:00:00

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/golem/scratch/chans/lincsv3
julia scripts/pretrain/exp_tf.jl

