#!/bin/bash
#SBATCH --job-name=part-benchmark
#SBATCH -A gts-vfung3
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -q inferno
#SBATCH --array=0-9

python benchmark_materials.py \
    --chunk_index $SLURM_ARRAY_TASK_ID \
    --chunk_size 10 \
    --material_list material_ids_rand_subset.csv \
    --num_atoms 1000000 \
    --output results/chunk_$SLURM_ARRAY_TASK_ID 