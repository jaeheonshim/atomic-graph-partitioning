#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4555

python error_test.py