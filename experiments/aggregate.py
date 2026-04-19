import argparse
import os
import json
import glob
try:
    import numpy as np
except ImportError:
    print("Error: numpy is required for aggregation. Run 'pip install numpy'")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Aggregate multiple seed results")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON used for runs")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    with open(args.config, 'r') as f:
        config = json.load(f)

    exp_name = config.get("name", os.path.basename(args.config).split('.')[0])
    raw_dir = "results/raw"
    
    # Find all matching result files
    pattern = os.path.join(raw_dir, f"{exp_name}_seed*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No result files found for experiment {exp_name} in {raw_dir}")
        return

    print(f"Found {len(files)} result files for {exp_name}.")
    
    all_metrics = []
    # Assuming history evaluations are aligned across runs
    all_histories = []
    evaluations_axis = None

    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
            all_metrics.append(data["metrics"])
            all_histories.append(data["history"]["objective_value"])
            if evaluations_axis is None:
                evaluations_axis = data["history"]["evaluations"]

    # Calculate aggregations
    final_objs = [m["final_objective"] for m in all_metrics]
    mean_final_obj = np.mean(final_objs)
    std_final_obj = np.std(final_objs)

    hist_array = np.array(all_histories)
    mean_history = np.mean(hist_array, axis=0).tolist()
    std_history = np.std(hist_array, axis=0).tolist()

    summary = {
        "experiment_name": exp_name,
        "num_seeds": len(files),
        "config": config,
        "aggregated_metrics": {
            "final_objective_mean": mean_final_obj,
            "final_objective_std": std_final_obj,
        },
        "aggregated_history": {
            "evaluations": evaluations_axis,
            "objective_value_mean": mean_history,
            "objective_value_std": std_history
        }
    }

    summary_dir = "results/summary"
    os.makedirs(summary_dir, exist_ok=True)
    out_file = os.path.join(summary_dir, f"{exp_name}_summary.json")
    
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Aggregated summary saved to {out_file}")

if __name__ == "__main__":
    main()