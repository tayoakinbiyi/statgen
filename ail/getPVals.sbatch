#!/bin/bash
#SBATCH --job-name=ail
#SBATCH --output=slurm/getPVals.out
#SBATCH --error=slurm/getPVals.err
#SBATCH --partition=bigmem2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=19
#SBATCH --mem=60000

module load python/3.7.0
python /home/akinbiyi/ail/ail/getPVals.py