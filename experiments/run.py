import argparse
import json
import time
import os
from utils import save_raw_result

def run_experiment(config, seed):
    # This is a placeholder for the actual experiment dispatch
    # In Phase 2, we will route to specific synthetic benchmarks based on config["benchmark"]
    print(f"Running experiment '{config.get('name', 'unknown')}' with seed {seed}...")
    
    start_time = time.time()
    
    # MOCK RUN for infrastructure testing
    history = {
        "evaluations": [0, 100, 200, 300],
        "objective_value": [10.0, 5.0, 2.0, 0.5]
    }
    metrics = {
        "final_objective": 0.5,
        "total_evaluations": 300,
        "wall_clock_time": time.time() - start_time
    }
    
    return history, metrics

def main():
    parser = argparse.ArgumentParser(description="DGE Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    with open(args.config, 'r') as f:
        config = json.load(f)

    exp_name = config.get("name", os.path.basename(args.config).split('.')[0])
    
    history, metrics = run_experiment(config, args.seed)
    
    out_file = save_raw_result(exp_name, args.seed, config, history, metrics)
    print(f"Result saved to {out_file}")

if __name__ == "__main__":
    main()