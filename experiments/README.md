# DGE Experiments Infrastructure

This folder contains the standardized execution environment for all DGE benchmarks and experiments, ensuring reproducibility across seeds and configurations.

## Directory Structure

- `run.py`: The main entrypoint for executing an experiment from a configuration file.
- `aggregate.py`: Utility to combine results from multiple seeds into a single summary file.
- `plot.py`: Utility to generate plots with mean and variance bands from aggregated results.
- `utils.py`: Helper functions for logging hardware info, git commits, and managing result directories.
- `configs/`: Directory containing JSON configuration files defining the experiments.

The execution of these scripts will automatically generate a `results/` folder in the project root:
- `results/raw/`: Individual run logs (one per seed).
- `results/summary/`: Aggregated results across multiple seeds.
- `results/figures/`: Generated plots.

## How to run an experiment

1. Create or select a configuration file in `configs/`, e.g., `configs/sphere_d32.json`.
2. Run the experiment for a specific seed:
   ```bash
   python experiments/run.py --config experiments/configs/sphere_d32.json --seed 42
   ```
   Or use a bash script/loop to run multiple seeds.

3. Aggregate the results:
   ```bash
   python experiments/aggregate.py --config experiments/configs/sphere_d32.json
   ```

4. Plot the results:
   ```bash
   python experiments/plot.py --summary results/summary/sphere_d32_summary.json
   ```
