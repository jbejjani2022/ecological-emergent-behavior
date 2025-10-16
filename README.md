![alt text](https://github.com/jbejjani2022/ecological-emergent-behavior/blob/main/images/fig1.png?raw=true)

# Ecological Emergent Behavior
This is the experimental code for "The Emergence of Complex Behavior in Large-Scale Ecological Environments."

## Requirements
This repository uses the [dirt](http://www.github.com/aaronwalsman/dirt) ecological gridworld environment and [mechagogue](http://www.github.com/aaronwalsman/mechagogue) training tools.
These are not yet in PyPI, but are coming soon.  The dirt interactive 3D visualizer also requires [Splendor Render](http://www.github.com/aaronwalsman/splendor-render)
which rquires a few installation steps (see the dirt readme for details).

## Experiments
Below are instructions to rerun the experiments in the paper.
### 4.1 Long-Distance Resource Gathering

### 4.2 Visual Foraging and Predation

## Note on Reproducibility
Due to GPU non-determinism, we have found that using the same random seed does not guarantee identical behavior across multiple runs of our experiments.
Therefore, when running this code, you will almost certainly not reproduce the exact same data/plots that are in the paper, however the overall trends will be largely similar.
