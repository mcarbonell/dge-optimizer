import argparse
import os
import json

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required for plotting. Run 'pip install matplotlib numpy'")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Plot aggregated experiment results")
    parser.add_argument("--summaries", type=str, nargs='+', required=True, help="Paths to summary JSONs")
    parser.add_argument("--out", type=str, default="comparison_plot.png", help="Output filename")
    parser.add_argument("--title", type=str, default="Benchmark Comparison", help="Plot title")
    parser.add_argument("--metric", type=str, default="objective_value", help="Which metric from history to plot (e.g. objective_value, test_accuracy)")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for Y axis")
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))

    for summary_file in args.summaries:
        if not os.path.exists(summary_file):
            print(f"Error: Summary file {summary_file} not found. Skipping.")
            continue

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        exp_name = summary["experiment_name"]
        evals = np.array(summary["aggregated_history"]["evaluations"])
        
        mean_key = f"{args.metric}_mean"
        std_key = f"{args.metric}_std"
        
        if mean_key not in summary["aggregated_history"]:
            print(f"Warning: Metric {mean_key} not found in {summary_file}. Skipping.")
            continue
            
        means = np.array(summary["aggregated_history"][mean_key])
        stds = np.array(summary["aggregated_history"][std_key])

        p = plt.plot(evals, means, label=f'{exp_name} ({summary["num_seeds"]} seeds)')
        color = p[0].get_color()
        plt.fill_between(evals, means - stds, means + stds, color=color, alpha=0.2)

    if args.log_scale:
        plt.yscale('log')
        plt.ylabel(f'{args.metric} (Log Scale)')
    else:
        plt.ylabel(args.metric)
        
    plt.xlabel('Function Evaluations')
    plt.title(args.title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    figures_dir = "results/figures"
    os.makedirs(figures_dir, exist_ok=True)
    out_file = os.path.join(figures_dir, args.out)
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()