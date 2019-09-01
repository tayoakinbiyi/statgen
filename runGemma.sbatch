#!/bin/bash
#SBATCH --job-name=ail
#SBATCH --output=ail.out
#SBATCH --error=ail.err
#SBATCH --partition=bigmem2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150000

module load python/3.7.0
python /home/akinbiyi/ail/runGemma.py