"""
This module contains functions for making reports for the simulation.
"""

from typing import Any
import jax.numpy as jnp

from mechagogue.static import static_data


def make_blank_report(env, make_video, report_visualizer_data, report_family_tree, report_homicides):
    @static_data
    class Report:
        if make_video:
            video_frames: Any = False
        
        if report_visualizer_data:
            visualizer_report : Any = env.default_visualizer_report()
        
        if report_family_tree:
            family_tree : Any = env.empty_family_tree_state()
        
        if report_homicides:
            homicides : int = 0
            attacks : int = 0
            population : int = 0
    
    return Report


def make_reporting(Report, env, make_video, report_visualizer_data, report_family_tree, report_homicides):
    def make_report(state, players, parents, children, actions, traits):
        report = Report()
        if make_video:
            report = report.replace(
                video_frames = env.make_video_report(state.env_state))
        if report_visualizer_data:
            report = report.replace(
                visualizer_report = env.make_visualizer_report(
                    state.env_state, actions),
            )
        elif report_family_tree:
            report = report.replace(
                family_tree=state.env_state.bugs.family_tree)
        if report_homicides:
            report = report.replace(
                homicides=jnp.sum(state.env_state.homicides),
                attacks=jnp.sum(state.env_state.attacks),
                population=jnp.sum(env.active_players(state.env_state)),
            )
        
        return report
    
    return make_report
