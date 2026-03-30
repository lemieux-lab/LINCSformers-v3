#!/bin/bash

#SBATCH --job-name=rtf_50_trt
#SBATCH --output=out/2026-03-21/%x.log
#SBATCH --error=out/2026-03-21/%x.log

#SBATCH --partition=rhel9-aarch64

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=600G

#SBATCH --gres=gpu:GH200:1
#SBATCH --time=4-00:00:00

export JULIAUP_DEPOT_PATH="$HOME/.juliaup-aarch64"
export JULIA_DEPOT_PATH="$HOME/.julia-aarch64" 
export PATH="$HOME/.juliaup-bin-aarch64/bin:$HOME/.local-aarch64/bin:$JULIAUP_DEPOT_PATH/bin:$PATH"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/golem/scratch/chans/lincsv3
julia scripts/pretrain/rank_tf.jl
