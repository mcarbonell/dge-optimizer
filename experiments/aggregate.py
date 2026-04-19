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
    
    # We will aggregate any key found in metrics and history
    all_metrics = []
    all_histories = []
    
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
            all_metrics.append(data.get("metrics", {}))
            all_histories.append(data.get("history", {}))

    aggregated_metrics = {}
    if all_metrics:
        keys = all_metrics[0].keys()
        for k in keys:
            vals = [m[k] for m in all_metrics if k in m and isinstance(m[k], (int, float))]
            if vals:
                aggregated_metrics[f"{k}_mean"] = float(np.mean(vals))
                aggregated_metrics[f"{k}_std"] = float(np.std(vals))

    aggregated_history = {}
    if all_histories:
        keys = all_histories[0].keys()
        for k in keys:
            if k == "evaluations":
                aggregated_history[k] = all_histories[0][k]
            else:
                arrays = [h[k] for h in all_histories if k in h]
                if arrays:
                    arr = np.array(arrays)
                    aggregated_history[f"{k}_mean"] = np.mean(arr, axis=0).tolist()
                    aggregated_history[f"{k}_std"] = np.std(arr, axis=0).tolist()

    summary = {
        "experiment_name": exp_name,
        "num_seeds": len(files),
        "config": config,
        "aggregated_metrics": aggregated_metrics,
        "aggregated_history": aggregated_history
    }

    summary_dir = "results/summary"
    os.makedirs(summary_dir, exist_ok=True)
    out_file = os.path.join(summary_dir, f"{exp_name}_summary.json")
    
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Aggregated summary saved to {out_file}")

if __name__ == "__main__":
    main()