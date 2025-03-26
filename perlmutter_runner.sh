#!/bin/bash
#SBATCH -C gpu
#SBATCH -A m4555
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

python error_test.py