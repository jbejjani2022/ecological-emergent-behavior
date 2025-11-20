"""
Parameter sweep script for running multiple simulation experiments.

This script handles hyperparameter sweeps by launching multiple train.py processes
with different world sizes, network architectures, and environmental configurations.
It's useful for systematic exploration of parameter spaces and comparative studies.
"""

import argparse
import os
import shlex
import subprocess
import sys
import warnings
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
# Search from current file's directory up to find .env
load_dotenv(find_dotenv())

def main():
    parser = argparse.ArgumentParser(description="Run sweep over world and network sizes for a given environment")
    parser.add_argument("--output_dir", type=str, default="./out", help="Path to output directory")
    parser.add_argument("--log_wandb", type=int, default=1, choices=[0, 1], help="1 for logging to wandb, 0 for no logging")
    parser.add_argument("--make_epoch_images", type=int, default=1, choices=[0, 1], help="1 for making epoch images, 0 for no epoch images")
    parser.add_argument("--make_video", type=int, default=0, choices=[0, 1], help="1 for making video, 0 for no video")
    parser.add_argument("--vision", type=int, default=0, choices=[0, 1], help="1 for vision, 0 for zero vision")
    parser.add_argument("--compass", type=int, choices=[0, 1], required=True, help="1 for compass, 0 for no compass")
    parser.add_argument("--violence", type=int, choices=[0, 1], required=True, help="1 for violence, 0 for no violence")
    parser.add_argument(
        "--env",
        type=str,
        choices=["ocean", "slope", "isthmus", "channel", "island", "lake", "fractal"],
        required=True,
        help="Environment to run: ocean | slope | isthmus | channel | island | lake | fractal",
    )
    parser.add_argument("--world_sizes", type=int, nargs="+", required=True, help="World grid sizes")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_mutation_rate", type=float, default=3e-2)
    parser.add_argument("--initial_population", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name for wandb. If not provided, defaults to {env_name}-sweep")
    args = parser.parse_args()

    zero_vision = not args.vision
    compass_on = args.compass
    violence_on = args.violence
    seed = args.seed
    env_name = args.env
    out_dir_path = args.output_dir
    log_wandb = args.log_wandb
    make_epoch_images = args.make_epoch_images
    make_video = args.make_video
    epochs = args.epochs
    base_mutation_rate = args.base_mutation_rate
    initial_population = args.initial_population
    experiment_name_arg = args.experiment_name
    
    # Load wandb_entity from environment if log_wandb is true
    wandb_entity = None
    if log_wandb:
        wandb_entity = os.getenv('WANDB_ENTITY')
        if not wandb_entity:
            warnings.warn("WANDB_ENTITY environment variable not found.")
    
    # Variables to sweep
    world_sizes = args.world_sizes
    network_size = {"backbone_layers": 2, "hidden_channels": 64}

    env_flag_map = {
        "ocean": {
            "rock_mode": "constant",
            "rock_bias": -10.,
        },
        "fractal": {"rock_mode": "fractal"},
        "slope": {"rock_mode": "slope"},
        "isthmus": {"rock_mode": "isthmus"},
        "channel": {"rock_mode": "channel"},
        "island": {"rock_mode": "island"},
        "lake": {"rock_mode": "lake"},
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(script_dir, "train.py")

    def world_to_arg(size):
        return f"{size},{size}"

    for world_size in world_sizes:
        experiment_name = experiment_name_arg if experiment_name_arg is not None else f"{env_name}-sweep"
        landscape_seed = 2
        
        if world_size == 1024:  # i.e. 1024x1024
            initial_players = 32768
            max_players = 131072
        elif world_size == 512:
            initial_players = 8192
            max_players = 32768
        elif world_size == 256:
            initial_players = 2048
            max_players = 16384
        elif world_size == 128:
            initial_players = 512
            max_players = 8192
        elif world_size == 64:
            initial_players = 128
            max_players = 4096
            
        # override initial_players if it was given via command line
        if initial_population is not None:
            initial_players = initial_population
        
        layers, channels = network_size["backbone_layers"], network_size["hidden_channels"]
        vision_tag = "zero_vision" if zero_vision else "vision"
        compass_tag = "compass_on" if compass_on else "compass_off"

        run_name = f"{base_mutation_rate}_{initial_players}_{env_name}_{world_size}_{vision_tag}_{compass_tag}_{seed}"
        output_directory = os.path.join(
            out_dir_path,
            env_name,
            str(world_size),
            vision_tag,
            compass_tag,
            run_name,
        )

        cmd = [
            sys.executable,
            "-u",
            train_py,
            "--seed",
            str(seed),
            "--output_directory",
            output_directory,
            "--run_name",
            run_name,
            "--experiment_name",
            experiment_name,
            "--log_wandb",
            str(int(log_wandb)),
            "--make_epoch_images",
            str(int(make_epoch_images)),
            "--make_video",
            str(int(make_video)),
            "--epochs",
            str(epochs),
            "--initial_players",
            str(initial_players),
            "--world_size",
            world_to_arg(world_size),
            "--max_players",
            str(max_players),
            "--hidden_channels",
            str(channels),
            "--backbone_layers",
            str(layers),
            "--zero_vision",
            str(int(zero_vision)),
            "--include_compass",
            str(int(compass_on)),
            "--include_violence",
            str(int(violence_on)),
            "--landscape_seed",
            str(landscape_seed),
            "--model_params-base_mutation_rate",
            str(base_mutation_rate)
        ]
        
        # Add wandb_entity if it's set
        if wandb_entity:
            cmd.extend(["--wandb_entity", wandb_entity])

        # Append environment-specific flags
        env_flags = env_flag_map[env_name]
        for key, value in env_flags.items():
            cmd.append(f"--{key}")
            if isinstance(value, bool):
                cmd.append("1" if value else "0")
            else:
                cmd.append(str(value))

        print("\nLaunching:")
        print(" ", shlex.join(cmd))
        env = os.environ.copy()
        # Get actual project root: experiments/ -> ecological_emergent_behavior/ -> project_root/
        package_dir = os.path.dirname(script_dir)
        project_root = os.path.dirname(package_dir)
        env['PYTHONPATH'] = project_root + ':' + env.get('PYTHONPATH', '')
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
