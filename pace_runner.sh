#!/bin/bash
#SBATCH -J MattersimRunner
#SBATCH -A gts-vfung3
#SBATCH -N 1 --gres=gpu:H200:1
#SBATCH -q inferno
#SBATCH -t 8:00:00
#SBATCH --mem-per-gpu=200G

python error_test_old.py