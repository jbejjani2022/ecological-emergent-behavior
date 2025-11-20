#!/bin/bash
#SBATCH --job-name=chase_runs_512
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-9

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate dirt

for env in slope; do
  for compass in 1 0; do
    world_sizes=(512)

    python -m ecological_emergent_behavior.experiments.sweep \
      --output_dir /n/netscratch/sham_lab/Everyone/jbejjani/ecological_emergent_behavior/ablations \
      --env "$env" \
      --world_sizes "${world_sizes[@]}" \
      --compass "$compass" \
      --vision 0 \
      --violence 0 \
      --epochs 400 \
      --make_epoch_images 0 \
      --make_video 0 \
      --seed "$SLURM_ARRAY_TASK_ID" \
      --experiment_name "chase-slope-sweep"
  done
done
