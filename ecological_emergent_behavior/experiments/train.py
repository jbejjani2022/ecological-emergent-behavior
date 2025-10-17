"""
Single simulation entry point with configurable parameters.

This script sets up and runs a single ecological simulation experiment with specific
model and environment configurations. It can be run standalone or called by sweep.py
for parameter sweeps. Command-line arguments override default parameters via TrainParams.
"""

from ecological_emergent_behavior.utils.xla import apply_xla_flags
apply_xla_flags()

import jax.numpy as jnp

from dirt.envs.tera_arium import TeraAriumParams
from dirt.gridworld2d.landscape import LandscapeParams
from dirt.bug import BugParams

from ecological_emergent_behavior.experiments.run_simulation import run_simulation, TrainParams
from ecological_emergent_behavior.models.bug_model import BugModelParams


model_params = BugModelParams(
    vision_encoder_mode='flatten',
    include_backbone=True,
    backbone_mode='mlp',
    mutate_traits=False,
    sensors_start_noisy=False,
    mutate_sensor_noise=False,
    mutate_color=True,
    base_mutation_rate=3e-2,
    learnable_temperature=True,
)


if __name__ == '__main__':
    params = TrainParams(
        model_params=model_params,
        save_states=False,
        save_reports=False,
        env_params=TeraAriumParams(
            landscape=LandscapeParams(
                energy_fractal_mask=False,
                biomass_fractal_mask=False,
                include_water_sources_and_sinks=False,
                initial_biomass_site_density = 1./4.,
            ),
            bugs=BugParams(
                include_expell_actions=False,
            ),
            display_hit_map=True,
        ),
        include_vision_encoder=True,
        include_water=True,
        include_rain=False,
        include_wind=False,
        include_temperature=False,
        include_light=False,
        include_rock=True,
        include_audio=False,
        include_smell=False,
        vision_includes_rgb=True,
        vision_includes_relative_altitude=True,
        max_view_width=7,
        max_view_distance=3,
        max_view_back_distance=3,
    ).from_commandline()

    run_simulation(params, model_float_dtype=jnp.bfloat16)
