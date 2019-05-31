#!/bin/bash
#SBATCH --job-name=genCorr
#SBATCH --output=slurm/genCorr.out
#SBATCH --error=slurm/genCorr.err
#SBATCH --partition=broadwl-lc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150000

module load python/3.7.0
python /home/akinbiyi/ail/ail/genCorr.py > /home/akinbiyi/ail/ail/gc