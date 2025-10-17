"""
Core simulation logic for ecological emergent behavior experiments.

This module defines the TrainParams configuration class and run_simulation() function
that orchestrate the main simulation loop. It handles environment setup, population 
initialization, natural selection, logging, and epoch progression.
"""

import os
from typing import Tuple
from dataclasses import asdict
from dotenv import load_dotenv, find_dotenv

import wandb
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.commandline import commandline_interface
from mechagogue.static import static_data
from mechagogue.epoch import make_epoch_system
from mechagogue.ecology.natural_selection import NaturalSelectionParams, make_natural_selection
from mechagogue.serial import save_leaf_data, load_example_data

from dirt.constants import DEFAULT_FLOAT_DTYPE
from dirt.envs.tera_arium import TeraAriumParams, make_tera_arium

from ecological_emergent_behavior.models.bug_model import BugModelParams, make_bug_population

from ecological_emergent_behavior.utils.log import make_logger
from ecological_emergent_behavior.utils.report import make_reporting, make_blank_report

# Load environment variables from .env file
# Search from current file's directory up to find .env
load_dotenv(find_dotenv())

@commandline_interface(override_descendants=True)
@static_data
class TrainParams:
    # training params
    experiment_name : str = 'ecological-emergent-behavior'
    run_name : str = 'base'
    wandb_entity : str = os.getenv('WANDB_ENTITY')
    seed : int = 1
    output_directory : str = '.'
    load_epoch : int = 0
    use_loaded_key : bool = True
    epochs : int = 200
    steps_per_epoch : int = 5000
    save_states : bool = True
    save_reports : bool = False
    verbose : bool = True
    stop_when_extinct : bool = True
    log_wandb : bool = True
    
    # environment and natural selection params
    initial_players : int = 8192
    max_players : int = 8192
    world_size : Tuple[int,int] = (1024,1024)
    env_params : TeraAriumParams = TeraAriumParams()
    natural_selection_params : NaturalSelectionParams = NaturalSelectionParams()
    
    # high level feature settings
    include_rock : bool = True
    include_water : bool = True
    include_energy : bool = True
    include_biomass : bool = True
    include_wind : bool = True
    include_temperature : bool = True
    include_rain : bool = True
    include_light : bool = True
    include_audio : bool = True
    audio_channels : int = 8
    include_smell : bool = True
    smell_channels : int = 8
    include_compass : int = True
    vision_includes_rgb : bool = True
    vision_includes_relative_altitude : bool = True
    
    include_violence : bool = False
    include_expell_actions : bool = False
    
    rock_mode : str = 'fractal'
    landscape_seed : int = None
    rock_bias : float = 0
    
    # model params
    hidden_channels : int = 64
    backbone_layers : int = 2
    include_vision_encoder : bool = False
    zero_vision : bool = False
    max_view_width : int = 15
    max_view_distance : int = 7
    max_view_back_distance : int = 7
    model_params : BugModelParams = BugModelParams()
    
    # reporting params
    report_bug_actions : bool = False
    report_bug_internals : bool = False
    report_bug_traits : bool = False
    report_object_grid : bool = False
    report_visualizer_data : bool = True
    report_family_tree : bool = True
    report_homicides : bool = True
    
    # visualization params
    make_video : bool = False
    make_epoch_images : bool = False
    visualize : bool = False
    window_size : Tuple[int, int] = (512,512)


def run_simulation(
    params: TrainParams,
    train_float_dtype = DEFAULT_FLOAT_DTYPE,
    model_float_dtype = DEFAULT_FLOAT_DTYPE,
):
    # make the output directory
    if not os.path.exists(params.output_directory):
        os.makedirs(params.output_directory)
        
    params_path = f'{params.output_directory}/params.data'
    
    # make key
    key = jrng.key(params.seed)
    
    # build the environment
    env = make_tera_arium(params.env_params, train_float_dtype)
    
    # build the population
    population = make_bug_population(
        params.model_params, env, model_float_dtype)
    
    # build the trainer
    natural_selection = make_natural_selection(
        params.natural_selection_params,
        env,
        population,
    )
    
    # set up the reporting
    Report = make_blank_report(env, params.make_video, params.report_visualizer_data, params.report_family_tree, params.report_homicides)
    make_report = make_reporting(Report, env, params.make_video, params.report_visualizer_data, params.report_family_tree, params.report_homicides)
    
    # make the logger
    log = make_logger(env, params.env_params, population, params.model_params.backbone_mode, params.log_wandb, params.make_video, params.make_epoch_images, params.output_directory)
    
    # make the epoch system
    epoch_system = make_epoch_system(
        natural_selection,
        params.steps_per_epoch,
        make_report=make_report,
        log=log,
        output_directory=params.output_directory,
        save_states=params.save_states,
        save_reports=params.save_reports,
        verbose=params.verbose,
    )
    
    # visualize
    if params.visualize:
        reports_paths = sorted([
            f'{params.output_directory}/{file_path}'
            for file_path in os.listdir(params.output_directory)
            if file_path.startswith('report') and file_path.endswith('.data')
        ])
        
        from dirt.visualization.viewer import Viewer
        
        viewer = Viewer(
            Report(),
            reports_paths,
            params.world_size,
            window_size=params.window_size,
            get_report_block=lambda report : report.visualizer_report,
            get_terrain_map=env.visualizer_terrain_map,
            get_terrain_texture=env.visualizer_terrain_texture,
            get_water_map=None,
            get_sun_direction=None,
            print_player_info=env.print_player_info,
            terrain_texture_resolution=(1024,1024),
        )
        viewer.start()
    
    else:
        # initialize state
        key, init_key = jrng.split(key)
        
        if params.load_epoch:
            loaded_key, state = epoch_system.load_epoch(params.load_epoch)
            if params.use_loaded_key:
                key = loaded_key
        else:
            state = epoch_system.init(init_key)
        
        # setup wandb
        if params.log_wandb:
            wandb.init(
                project=params.experiment_name,
                name=params.run_name,
                entity=params.wandb_entity,
            )
            wandb.config.update(asdict(params))
        
        # perform an initial logging step
        key, log_key = jrng.split(key)
        log(log_key, state.epoch, state.system_state, None)
        
        # save params
        save_leaf_data(params, params_path)
        
        try:
            # run
            for i in range(int(params.load_epoch), params.epochs):
                key, step_key = jrng.split(key)
                state = epoch_system.step(step_key, state)
                
                if (
                    params.stop_when_extinct and
                    env.extinct(state.system_state.env_state)
                ):
                    print('extinct, exiting')
                    break
        finally:
            if params.save_reports and params.report_family_tree:
                reports_paths = sorted([
                    f'{params.output_directory}/{file_path}'
                    for file_path in os.listdir(params.output_directory)
                    if file_path.startswith('report')
                    and file_path.endswith('.data')
                ])
                if len(reports_paths):
                    all_players = []
                    all_parents = []
                    example_report = Report()
                    for reports_path in reports_paths:
                        print(
                            f'collecting family tree data from {reports_path}')
                        reports = load_example_data(
                            example_report, reports_path)
                        all_players.append(
                            reports.family_tree.player_state.players)
                        all_parents.append(reports.family_tree.parents)
                    
                    players = jnp.concatenate(all_players, axis=0)
                    parents = jnp.concatenate(all_parents, axis=0)
                    family_tree_path = (
                        f'{params.output_directory}/family_tree.data')
                    print(f'saving family tree data to {family_tree_path}')
                    save_leaf_data((players, parents), family_tree_path)
