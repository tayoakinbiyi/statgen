#!/bin/bash
#SBATCH --job-name=maPlots
#SBATCH --output=slurm/maPlots.out
#SBATCH --error=slurm/maPlots.err
#SBATCH --partition=bigmem2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60000

module load python/3.7.0
python /home/akinbiyi/ail/ail/maPlots.py