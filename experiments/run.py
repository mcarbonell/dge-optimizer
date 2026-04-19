import argparse
import json
import time
import os
import sys
import numpy as np

# Ensure root directory is in path to import dge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dge.optimizer import DGEOptimizer
from experiments.utils import save_raw_result
from experiments.benchmarks import get_benchmark
from experiments.baselines import SPSAOptimizer, RandomDirectionOptimizer

def run_experiment(config, seed):
    print(f"Running experiment '{config.get('name', 'unknown')}' with seed {seed}...")
    
    start_time = time.time()
    
    dim = config.get("dimension", 128)
    budget = config.get("budget", 10000)
    benchmark_name = config.get("benchmark", "sphere")
    f = get_benchmark(benchmark_name)
    
    rng = np.random.default_rng(seed)
    
    # Initialization
    init_range = config.get("init_range", [-5.0, 5.0])
    x = rng.uniform(init_range[0], init_range[1], size=dim).astype(np.float32)
    
    opt_config = config.get("optimizer", {})
    opt_name = opt_config.get("name", "dge")
    
    # Calculate total steps estimate roughly for schedules
    if opt_name == "dge":
        k = opt_config.get("k", max(1, int(np.ceil(np.log2(dim)))))
        total_steps = budget // (2 * k)
        opt_params = {k: v for k, v in opt_config.items() if k not in ["name", "k"]}
        opt = DGEOptimizer(dim=dim, seed=seed, total_steps=total_steps, **opt_params)
    elif opt_name == "spsa":
        total_steps = budget // 2
        opt_params = {k: v for k, v in opt_config.items() if k not in ["name"]}
        opt = SPSAOptimizer(dim=dim, seed=seed, total_steps=total_steps, **opt_params)
    elif opt_name == "random":
        total_steps = budget // 2
        opt_params = {k: v for k, v in opt_config.items() if k not in ["name"]}
        opt = RandomDirectionOptimizer(dim=dim, seed=seed, total_steps=total_steps, **opt_params)
    else:
        raise ValueError(f"Unknown optimizer {opt_name}")

    evals = 0
    history_evals = []
    history_obj = []
    
    # Record initial state
    obj_val = f(x)
    history_evals.append(evals)
    history_obj.append(float(obj_val))
    
    internal_time = 0.0
    f_time = 0.0

    # For tracking tracking periodicity
    log_interval = max(1, budget // 100)
    next_log = log_interval

    while evals < budget:
        t0 = time.time()
        
        # Wrapper to track f_time
        def tracked_f(x_in):
            nonlocal f_time
            t_f0 = time.time()
            v = f(x_in)
            f_time += time.time() - t_f0
            return v
            
        # Optimization step
        x, evals_used = opt.step(tracked_f, x)
        internal_time += time.time() - t0 - f_time
        
        evals += evals_used
        
        # Log periodically
        if evals >= next_log or evals >= budget:
            obj_val = f(x)
            history_evals.append(evals)
            history_obj.append(float(obj_val))
            next_log += log_interval

    total_time = time.time() - start_time
    
    history = {
        "evaluations": history_evals,
        "objective_value": history_obj
    }
    metrics = {
        "final_objective": float(history_obj[-1]),
        "total_evaluations": evals,
        "wall_clock_time": total_time,
        "internal_overhead_time": internal_time,
        "function_evaluation_time": f_time
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