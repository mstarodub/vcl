#!/bin/bash
#SBATCH --clusters=arc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=medium
#SBATCH --array=0-9

cd $DATA/vcl
uv run main.py
