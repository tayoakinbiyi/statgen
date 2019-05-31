#!/bin/bash
#SBATCH --job-name=qqPlots
#SBATCH --output=slurm/qqPlots.out
#SBATCH --error=slurm/qqPlots.err
#SBATCH --partition=bigmem2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60000

module load python/3.7.0
#python /home/akinbiyi/ail/ail/qqPlotsAll.py
python /home/akinbiyi/ail/ail/qqPlotsSmall.py