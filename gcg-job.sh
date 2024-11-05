#!/bin/bash

#SBATCH --job-name=gcg_eval_job
#SBATCH --gres=gpu:A100:3
#SBATCH --partition=gpu

srun --container-image=floofcat/pvt_advattacks:v5 --container-mounts=$HOME/CISPA-az6/adv_attacks_llm-2024/baseline:/root python3 $HOME/CISPA-az6/adv_attacks_llm-2024/baseline/gasp-acq/gcg.py 