#!/bin/bash
#SBATCH -J MattersimRunner
#SBATCH -A gts-vfung3
#SBATCH -N 1 --gres=gpu:H200:1
#SBATCH -q inferno
#SBATCH -t 4:00:00

python error_test.py