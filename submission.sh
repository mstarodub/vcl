#!/bin/bash
#SBATCH --clusters=arc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=long
#SBATCH --array=0-79

cd $DATA/vcl
uv run main.py
