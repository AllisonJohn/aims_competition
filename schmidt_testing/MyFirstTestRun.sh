#!/bin/bash
#SBATCH --job-name=MyFirstTestRun
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

srun nvidia-smi -q
