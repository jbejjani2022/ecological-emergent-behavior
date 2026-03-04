#!/bin/bash
#SBATCH --job-name=profiling_default
#SBATCH --account=kempner_awalsman_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --mem=256G
#SBATCH -o out/job.%N.%j.out          # STDOUT
#SBATCH -e out/job.%N.%j.err           # STDERR
#SBATCH --mail-type=ALL
#SBATCH --array=0

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate dirt

python -m ecological_emergent_behavior.experiments.sweep \
  --output_dir /n/netscratch/kempner_awalsman_lab/Everyone/awalsman/ecological_emergent_behavior \
  --env fractal \
  --world_sizes 256 \
  --tile_rows 1 \
  --tile_cols 4 \
  --compass 0 \
  --vision 1 \
  --violence 0 \
  --epochs 5 \
  --log_wandb 0 \
  --make_epoch_images 0 \
  --make_video 0 \
  --seed "$SLURM_ARRAY_TASK_ID" \
  --vision_range 7 3 3 \
  --initial_population 2048 \
  --max_population 16384 \
  --network_size 2 64 \
  --experiment_name "profiling-default"
  --model_params-policy_transfer_max_k 16
