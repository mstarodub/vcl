#!/bin/bash
#SBATCH --clusters=arc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=devel

cd $DATA
uv --version
