#!/bin/bash

# Phase 2 ERA5 smoke test. This intentionally trains for only 60k steps with
# one worker and one held-out scenario. Once the sampled corpus is validated,
# switch the manifest to london-spring.json, restore 12 eval scenarios, choose
# the production worker count, remove --timesteps to use the 15M default, and
# increase --time based on the measured smoke runtime. Rename the job/log from
# "smoke" as well so production output is unmistakable.

#SBATCH --job-name=Stratus_era5_smoke
#SBATCH --output=/scratch_tide/as5023/Stratus/MachineLearning/logs/slurm/era5-smoke-%j.log

#SBATCH --partition=tide
#SBATCH --qos=intermediate

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

source /scratch_tide/as5023/miniconda3/etc/profile.d/conda.sh
conda activate Stratus

start_time=$(date +%s)
echo -e "Phase 2 ERA5 smoke job started on $(date)\n"

cd /scratch_tide/as5023/Stratus/MachineLearning

weather_manifest="weather_data/manifests/london-smoke.json"
if [[ ! -f "$weather_manifest" ]]; then
    echo "Missing $weather_manifest"
    echo "Prepare it first with:"
    echo "  python scripts/acquire_era5_corpus.py --profile smoke --execute"
    exit 1
fi

python main.py \
    --train \
    --dim 3 \
    --balloon-type zero_pressure \
    --weather-manifest "$weather_manifest" \
    --n-envs 1 \
    --timesteps 60000 \
    --eval-scenarios 1 \
    --save_fig \
    --gpu \
    --hpc

end_time=$(date +%s)

echo -e "\nPhase 2 ERA5 smoke job finished on $(date)"

total_seconds=$((end_time - start_time))
total_minutes=$((total_seconds / 60))
remaining_seconds=$((total_seconds % 60))
echo "Total runtime: ${total_minutes} minutes and ${remaining_seconds} seconds"
