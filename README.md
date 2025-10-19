# Ecological Emergent Behavior
This is the experimental code for "The Emergence of Complex Behavior in Large-Scale Ecological Environments."

![alt text](https://github.com/jbejjani2022/ecological-emergent-behavior/blob/main/images/fig1.png?raw=true)
_A 3D visualization of a 512 × 512 map in our environment. Insert (a) shows a top
down view of the entire map, (b) shows a group of agents clustered in the corner, (c) shows
agents in the water, (d) shows agents travelling back and forth from the land and (e) shows
a close up of a single agent gathering resources._

## Requirements
This repository uses the [`dirt`](http://www.github.com/aaronwalsman/dirt) ecological gridworld environment and [`mechagogue`](http://www.github.com/aaronwalsman/mechagogue) training tools.
These are coming soon to PyPI. The [`dirt`](http://www.github.com/aaronwalsman/dirt) interactive 3D visualizer depends on [`splendor-render`](http://www.github.com/aaronwalsman/splendor-render),
which requires a few installation steps detailed in the `dirt` [README](http://www.github.com/aaronwalsman/dirt).

## Setup

1. Follow the instructions in the `dirt` [README](http://www.github.com/aaronwalsman/dirt) to set up the environment.

2. Run `pip install -e .` from the project root to install this package in editable mode.

## Experiments
Below are instructions to rerun the experiments in the paper, which were run on one H100 GPU.

For logging to Weights & Biases, create a `.env` file in the root directory with the following contents:
```bash
WANDB_ENTITY="your wandb entity"
```

### 4.1 Long-Distance Resource Gathering
If working on a cluster using the SLURM workload manager, set your account and partition in the header of `ecological_emergent_behavior/experiments/long_distance_resource_gathering.sh` and run the job from the project root:
```bash
sbatch ecological_emergent_behavior/experiments/long_distance_resource_gathering.sh
```

### 4.2 Visual Foraging and Predation
On a cluster with SLURM, set your account and partition in the header of `ecological_emergent_behavior/experiments/visual_foraging_and_predation.sh`, then run the job from the project root:
```bash
sbatch ecological_emergent_behavior/experiments/visual_foraging_and_predation.sh
```

### Custom Experiments
The `sweep.py` script can be used to configure and run your own experiments. After installing the package, you can run it from anywhere using Python's module syntax. It takes the following arguments:

#### Required
- `--env`: Environment to run (`ocean`, `slope`, `isthmus`, `channel`, `island`, `lake`, `fractal`)
- `--world_sizes`: One or more (square) gridworld sizes (e.g., `64`, `128`, `256`, `512`, `1024`)
- `--compass`: Enable compass (1) or disable (0)
- `--violence`: Enable attacking (1) or disable (0)

#### Optional
- `--output_dir`: Path to output directory (default: `./out`)
- `--log_wandb`: Log to Weights & Biases (default: 1)
- `--make_epoch_images`: Generate an image of the gridworld at each epoch (default: 1)
- `--make_video`: Generate video of each epoch (default: 0)
- `--vision`: Enable vision in bugs (default: 0)
- `--epochs`: Max number of epochs to run (default: 400)
- `--seed`: Random seed (default: 0)

#### Example Usage:
```bash
python -m ecological_emergent_behavior.experiments.sweep \
  --env slope \
  --world_sizes 256 512 \
  --compass 1 \
  --violence 0 \
  --vision 0 \
  --epochs 400 \
  --make_epoch_images 1 \
  --make_video 0 \
  --seed 0
```

This example runs experiments in the `slope` terrain with 2 different world sizes (256×256, 512×512), enables the compass sensor in bugs, disables attacking, uses zero vision, runs for 400 epochs, generates epoch images but no video, and uses seed 0.

## Note on Reproducibility
Due to GPU non-determinism, we have found that using the same random seed does not guarantee identical behavior across multiple runs of our experiments.
Therefore, when running this code, you will almost certainly not reproduce the exact same data/plots that are in the paper. However, the primary results will be largely similar.
