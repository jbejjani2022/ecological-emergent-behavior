#!/bin/bash
#SBATCH --job-name=predation_and_foraging_sweep
#SBATCH --account=your_account
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --array=0-3

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate dirt

for env in slope isthmus channel island lake; do
  for compass in 0 1; do
    if [[ "$env" == "slope" ]]; then
      world_sizes=(1024 512 256 128)
    else
      world_sizes=(512)
    fi
    python -m ecological_emergent_behavior.experiments.sweep \
      --env "$env" \
      --world_sizes "${world_sizes[@]}" \
      --compass "$compass" \
      --vision 0 \
      --violence 0 \
      --epochs 400 \
      --make_epoch_images 1 \
      --make_video 0 \
      --seed "$SLURM_ARRAY_TASK_ID"
  done
done
