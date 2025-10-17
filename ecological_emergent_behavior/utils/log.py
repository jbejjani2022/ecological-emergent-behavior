"""
This module contains a function for logging environment
and bug trait data to wandb (optionally).
"""

import os
import numpy as np
import wandb
import jax.numpy as jnp
import imageio as imageio

import dirt.gridworld2d.grid as grid


def make_logger(env, env_params, population, backbone_mode, log_wandb, make_video, make_epoch_images, output_directory):
    def log(key, epoch, state, reports):
        if log_wandb:
            datapoint = {}
            t = int(state.env_state.landscape.time)
            
            # log bug population size
            num_players = int(env.population(state.env_state))
            datapoint['general/population'] = num_players
            active = env.active_players(state.env_state)
            
            # log average population age
            age = state.env_state.bugs.age
            age_sum = jnp.sum(age, dtype=jnp.float32)
            age_mean = age_sum/num_players
            datapoint['general/age_mean'] = age_mean
            datapoint['general/age_max'] = jnp.max(age)
            
            # log average generation
            generation = state.env_state.bugs.generation
            generation = jnp.where(generation >= 0, generation, 0)
            generation_sum = jnp.sum(generation, dtype=jnp.float32)
            generation_mean = generation_sum / num_players
            datapoint['general/generation_mean'] = generation_mean
            
            # log joint action probs
            attack_actions = state.env_state.bugs.attack_actions
            eat_actions = state.env_state.bugs.eat_actions
            move_actions = state.env_state.bugs.move_actions
            reproduce_actions = state.env_state.bugs.reproduce_actions
            no_actions = state.env_state.bugs.no_actions
            
            z = (
                attack_actions +
                eat_actions +
                move_actions +
                reproduce_actions +
                no_actions
            ).astype(jnp.float32) + 1e-8
            
            per_agent_attack_frequency = attack_actions / z
            per_agent_eat_frequency = eat_actions / z
            per_agent_move_frequency = move_actions / z
            per_agent_reproduce_frequency = reproduce_actions / z
            per_agent_no_frequency = no_actions / z
            
            datapoint['action/per_agent_attack'] = jnp.sum(
                per_agent_attack_frequency) / num_players
            datapoint['action/per_agent_eat'] = jnp.sum(
                per_agent_eat_frequency) / num_players
            datapoint['action/per_agent_move'] = jnp.sum(
                per_agent_move_frequency) / num_players
            datapoint['action/per_agent_reproduce'] = jnp.sum(
                per_agent_reproduce_frequency) / num_players
            datapoint['action/per_agent_no_action'] = jnp.sum(
                per_agent_no_frequency) / num_players
            
            per_agent_min_attack_move = jnp.minimum(
                per_agent_attack_frequency, per_agent_move_frequency)
            datapoint['action/per_agent_min_attack_move_mean'] = jnp.sum(
                per_agent_min_attack_move) / num_players
            
            if num_players:   
                (
                    pa_min_attack_move_99,
                    pa_min_attack_move_95,
                    pa_min_attack_move_90,
                    pa_min_attack_move_75,
                ) = jnp.percentile(
                    per_agent_min_attack_move[active],
                    jnp.array([99, 95, 90, 75])
                )
                datapoint['action/per_agent_min_attack_move_99'] = (
                    pa_min_attack_move_99)
                datapoint['action/per_agent_min_attack_move_95'] = (
                    pa_min_attack_move_95)
                datapoint['action/per_agent_min_attack_move_90'] = (
                    pa_min_attack_move_90)
                datapoint['action/per_agent_min_attack_move_75'] = (
                    pa_min_attack_move_75)

            per_agent_attack_move_product = (
                per_agent_attack_frequency * per_agent_move_frequency)
            datapoint['action/per_agent_attack_move_product'] = jnp.sum(
                per_agent_attack_move_product) / num_players
            
            if num_players:
                (
                    pa_attack_move_product_99,
                    pa_attack_move_product_95,
                    pa_attack_move_product_90,
                    pa_attack_move_product_75,
                ) = jnp.percentile(
                    per_agent_attack_move_product[active],
                    jnp.array([99, 95, 90, 75])
                )
                datapoint['action/per_agent_attack_move_product_99'] = (
                    pa_attack_move_product_99)
                datapoint['action/per_agent_attack_move_product_95'] = (
                    pa_attack_move_product_95)
                datapoint['action/per_agent_attack_move_product_90'] = (
                    pa_attack_move_product_90)
                datapoint['action/per_agent_attack_move_product_75'] = (
                    pa_attack_move_product_75)

                (
                    pa_attack_99,
                    pa_attack_95,
                    pa_attack_90,
                    pa_attack_75,
                ) = jnp.percentile(
                    per_agent_attack_frequency[active],
                    jnp.array([99, 95, 90, 75])
                )
                datapoint['action/per_agent_attack_99'] = (
                    pa_attack_99)
                datapoint['action/per_agent_attack_95'] = (
                    pa_attack_95)
                datapoint['action/per_agent_attack_90'] = (
                    pa_attack_90)
                datapoint['action/per_agent_attack_75'] = (
                    pa_attack_75)
            
                (
                    pa_eat_99,
                    pa_eat_95,
                    pa_eat_90,
                    pa_eat_75,
                ) = jnp.percentile(
                    per_agent_eat_frequency[active],
                    jnp.array([99, 95, 90, 75])
                )
                datapoint['action/per_agent_eat_99'] = (
                    pa_eat_99)
                datapoint['action/per_agent_eat_95'] = (
                    pa_eat_95)
                datapoint['action/per_agent_eat_90'] = (
                    pa_eat_90)
                datapoint['action/per_agent_eat_75'] = (
                    pa_eat_75)
                
                (
                    pa_move_99,
                    pa_move_95,
                    pa_move_90,
                    pa_move_75,
                ) = jnp.percentile(
                    per_agent_move_frequency[active],
                    jnp.array([99, 95, 90, 75])
                )
                datapoint['action/per_agent_move_99'] = (
                    pa_move_99)
                datapoint['action/per_agent_move_95'] = (
                    pa_move_95)
                datapoint['action/per_agent_move_90'] = (
                    pa_move_90)
                datapoint['action/per_agent_move_75'] = (
                    pa_move_75)
                
                (
                    pa_reproduce_99,
                    pa_reproduce_95,
                    pa_reproduce_90,
                    pa_reproduce_75,
                ) = jnp.percentile(
                    per_agent_reproduce_frequency[active],
                    jnp.array([99, 95, 90, 75])
                )
                datapoint['action/per_agent_reproduce_99'] = (
                    pa_reproduce_99)
                datapoint['action/per_agent_reproduce_95'] = (
                    pa_reproduce_95)
                datapoint['action/per_agent_reproduce_90'] = (
                    pa_reproduce_90)
                datapoint['action/per_agent_reproduce_75'] = (
                    pa_reproduce_75)
            
            # log homicides
            if hasattr(reports, 'homicides'):
                datapoint['violence/total_homicides'] = jnp.mean(
                    reports.homicides, dtype=jnp.float32)
                datapoint['violence/homicides_per_player'] = jnp.mean(
                    reports.homicides.astype(jnp.float32) /
                    (reports.population.astype(jnp.float32) + 1e-6)
                )
                datapoint['violence/homicides_per_attack'] = jnp.mean(
                    reports.homicides.astype(jnp.float32) /
                    (reports.attacks.astype(jnp.float32) + 1e-6)
                )
                datapoint['violence/attacks_per_player'] = jnp.mean(
                    reports.attacks.astype(jnp.float32) /
                    (reports.population.astype(jnp.float32) + 1e-6)
                )
            
            # check and log water
            if state.env_state.landscape.water is not None:
                datapoint['water/landscape'] = jnp.sum(
                    state.env_state.landscape.water, dtype=jnp.float32)
                if jnp.any(state.env_state.landscape.water < 0.):
                    print('Warning: negative water detected in the landscape')
                mean_water = grid.grid_sum_to_mean(
                    state.env_state.landscape.water,
                    env_params.landscape.terrain_downsample)
                standing_water = (
                    mean_water > env_params.landscape.min_standing_water)
                total_cells = standing_water.shape[0] * standing_water.shape[1]
                percent_standing_water = jnp.sum(
                    standing_water, dtype=jnp.float32) / total_cells
                datapoint['water/standing_water'] = percent_standing_water
            else:
                datapoint['water/landscape'] = 0.
                datapoint['water/standing_water'] = 0.
            
            if state.env_state.bugs.water is not None:
                datapoint['water/bugs'] = jnp.sum(
                    state.env_state.bugs.water, dtype=jnp.float32)
                if jnp.any(state.env_state.bugs.water < 0.):
                    print('Warning: negative water detected in the bugs')
            else:
                datapoint['water/bugs'] = 0.
            
            # check and log moisture
            if state.env_state.landscape.moisture is not None:
                datapoint['water/moisture'] = jnp.sum(
                    state.env_state.landscape.moisture, dtype=jnp.float32)
                if jnp.any(state.env_state.landscape.moisture < 0.):
                    print('Warning: negative moisture detected')
            else:
                datapoint['water/moisture'] = 0.
            
            datapoint['water/total'] = (
                datapoint['water/landscape'] +
                datapoint['water/bugs'] +
                datapoint['water/moisture']
            )
            
            # check and log energy
            datapoint['energy/landscape'] = jnp.sum(
                state.env_state.landscape.energy, dtype=jnp.float32)
            datapoint['energy/landscape_max'] = jnp.max(
                state.env_state.landscape.energy)
            datapoint['energy/bugs'] = jnp.sum(
                state.env_state.bugs.energy, dtype=jnp.float32)
            datapoint['energy/total'] = (
                datapoint['energy/landscape'] +
                datapoint['energy/bugs']
            )
            if state.env_state.landscape.water is not None:
                dry_energy = grid.scale_grid(
                    state.env_state.landscape.energy, ~standing_water)
                wet_energy = grid.scale_grid(
                    state.env_state.landscape.energy, standing_water)
                datapoint['energy/free_dry'] = jnp.sum(
                    dry_energy, dtype=jnp.float32)
                datapoint['energy/free_wet'] = jnp.sum(
                    wet_energy, dtype=jnp.float32)
            if jnp.any(state.env_state.landscape.energy < 0.):
                print('Warning: negative energy detected in the landscape')
            if jnp.any(state.env_state.bugs.energy < 0.):
                print('Warning: negative energy detected in the bugs')
            
            # check and log biomass
            datapoint['biomass/landscape'] = jnp.sum(
                state.env_state.landscape.biomass, dtype=jnp.float32)
            datapoint['biomass/landscape_max'] = jnp.max(
                state.env_state.landscape.biomass)
            datapoint['biomass/bugs'] = jnp.sum(
                state.env_state.bugs.biomass, dtype=jnp.float32)
            datapoint['biomass/utilization'] = (
                datapoint['biomass/bugs'] / 
                (datapoint['biomass/bugs'] + datapoint['biomass/landscape']))
            datapoint['biomass/total'] = (
                datapoint['biomass/landscape'] +
                datapoint['biomass/bugs']
            )
            if state.env_state.landscape.water is not None:
                dry_biomass = grid.scale_grid(
                    state.env_state.landscape.biomass, ~standing_water)
                wet_biomass = grid.scale_grid(
                    state.env_state.landscape.biomass, standing_water)
                datapoint['biomass/free_dry'] = jnp.sum(
                    dry_biomass, dtype=jnp.float32)
                datapoint['biomass/free_wet'] = jnp.sum(
                    wet_biomass, dtype=jnp.float32)
            if jnp.any(state.env_state.landscape.biomass < 0.):
                print('Warning: negative biomass detected in the landscape')
            if jnp.any(state.env_state.bugs.biomass < 0.):
                print('Warning: negative biomass detected in the bugs')
            
            # log temperature
            if state.env_state.landscape.temperature is not None:
                mean_temperature = state.env_state.landscape.temperature.mean()
                mean_temperature /= (
                    env_params.landscape.temperature_downsample**2)
                datapoint['temperature/mean'] = mean_temperature
            
            # log how many bugs are on land
            if state.env_state.landscape.water is not None:
                if num_players == 0:
                    land_dwellers = 0.
                    bug_dry_energy = 0.
                    bug_wet_energy = 0.
                    bug_dry_biomass = 0.
                    bug_wet_biomass = 0.
                else:
                    bug_x = state.env_state.bugs.x
                    bug_standing_water = grid.read_grid_locations(
                        standing_water,
                        bug_x,
                        env_params.landscape.terrain_downsample,
                        downsample_scale=False,
                    )
                    num_not_in_water = jnp.sum(
                        (~bug_standing_water) * active, dtype=jnp.float32)
                    land_dwellers = (num_not_in_water / num_players)
                    
                    active_bug_energy = state.env_state.bugs.energy * active
                    bug_dry_energy = jnp.sum(
                        active_bug_energy * ~bug_standing_water,
                        dtype=jnp.float32
                    )
                    bug_wet_energy = jnp.sum(
                        active_bug_energy * bug_standing_water,
                        dtype=jnp.float32
                    )
                    
                    active_bug_biomass = state.env_state.bugs.biomass * active
                    bug_dry_biomass = jnp.sum(
                        active_bug_biomass * ~bug_standing_water,
                        dtype=jnp.float32
                    )
                    bug_wet_biomass = jnp.sum(
                        active_bug_biomass * bug_standing_water,
                        dtype=jnp.float32
                    )
                datapoint['general/land_dwellers'] = land_dwellers
                datapoint['biomass/bug_dry'] = bug_dry_biomass
                datapoint['biomass/bug_wet'] = bug_wet_biomass
                datapoint['energy/bug_dry'] = bug_dry_energy
                datapoint['energy/bug_wet'] = bug_wet_energy
            
            # log biomass requirement
            biomass_req = env.biomass_requirement(state.env_state.bug_traits)
            biomass_req = biomass_req[active]
            biomass_req_mean = float(jnp.mean(biomass_req))
            datapoint['traits/biomass_req_mean'] = biomass_req_mean
            
            datapoint.update(population.log(key, state.population_state, state.obs, active))

            wandb.log(datapoint, step=t)
        
        if make_video and reports is not None:
            video_frames = np.array(reports.video_frames).astype(np.uint8)
            # flip the y-axis
            video_frames = video_frames[:,::-1]
            video_dir = f'{output_directory}/videos'
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            video_path = f'{video_dir}/video_{epoch:08d}.mp4'
            writer = imageio.get_writer(
                video_path,
                fps=30,
                codec='libx264',
                ffmpeg_params=[
                    '-crf', '18',
                    '-preset', 'slow',
                ],
            )
            with writer:
                n = video_frames.shape[0]
                for i in range(n):
                    writer.append_data(video_frames[i])
        
        if make_epoch_images:
            epoch_image = env.make_video_report(state.env_state)
            # flip the y-axis
            epoch_image = epoch_image[::-1]
            image_dir = f'{output_directory}/images'
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            image_path = f'{image_dir}/image_{epoch:08d}.png'
            imageio.imwrite(image_path, epoch_image)
    
    return log
