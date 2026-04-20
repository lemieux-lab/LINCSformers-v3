#!/bin/bash

#SBATCH --account=def-lemieux1
#SBATCH --job-name=rank_tf
#SBATCH --output=out/rank_tf.out
#SBATCH --error=err/rank_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=124G
#SBATCH --gpus=h100:1

#SBATCH --time=0-6:00:00
#SBATCH --mail-user=serenaktc@gmail.com
#SBATCH --mail-type=ALL

if [ -z "$SLURM_SUBMIT_DIR" ]; then
    export SLURM_SUBMIT_DIR=$(pwd)
fi

cd $SLURM_SUBMIT_DIR

module load julia/1.11.3 cuda/12.2 cudnn
julia --project=/home/chans/links/scratch/lincsv3 /home/chans/links/scratch/lincsv3/scripts/rank_tf.jl