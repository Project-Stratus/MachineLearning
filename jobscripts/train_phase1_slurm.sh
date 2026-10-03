#!/bin/bash

# Phase 1: analytic wind field, no ERA5 weather manifest.

#SBATCH --job-name=Stratus_phase1
#SBATCH --output=/scratch_tide/as5023/Stratus/MachineLearning/logs/slurm/phase1-train-%j.log

#SBATCH --partition=tide
#SBATCH --qos=intermediate

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --mem=64G
#SBATCH --time=10:00:00

set -euo pipefail

source /scratch_tide/as5023/miniconda3/etc/profile.d/conda.sh
conda activate Stratus

start_time=$(date +%s)
echo -e "Phase 1 job started on $(date)\n"

cd /scratch_tide/as5023/Stratus/MachineLearning

python main.py --train --dim 3 --balloon-type zero_pressure -sf -g --hpc

end_time=$(date +%s)

echo -e "\nPhase 1 job finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))
echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
