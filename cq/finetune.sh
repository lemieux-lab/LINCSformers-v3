#!/bin/bash

#SBATCH --account=def-lemieux1
#SBATCH --job-name=E2E_1_rtf
#SBATCH --output=cq/out/2026-04-20/%x.log
#SBATCH --error=cq/out/2026-04-20/%x.log

#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --gpus-per-node=h100:1

#SBATCH --time=0-12:00
#SBATCH --mail-user=serenaktc@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -z "$SLURM_SUBMIT_DIR" ]; then
    export SLURM_SUBMIT_DIR=$(pwd)
fi

cd $SLURM_SUBMIT_DIR

module load julia cuda/12.9
julia --project=/home/chans/links/scratch/lincsv4/cq /home/chans/links/scratch/lincsv4/scripts/finetune/main_tf_cq.jl