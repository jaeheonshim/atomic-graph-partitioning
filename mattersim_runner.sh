#!/bin/bash
#SBATCH -J MattersimRunner
#SBATCH -A gts-vfung3
#SBATCH -N 1 --gres=gpu:A100:1
#SBATCH -q inferno
#SBATCH --mem-per-gpu=80G

python mattersim_test.py