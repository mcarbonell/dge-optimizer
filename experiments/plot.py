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
    parser.add_argument("--summary", type=str, required=True, help="Path to summary JSON")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"Error: Summary file {args.summary} not found.")
        return

    with open(args.summary, 'r') as f:
        summary = json.load(f)

    exp_name = summary["experiment_name"]
    evals = np.array(summary["aggregated_history"]["evaluations"])
    means = np.array(summary["aggregated_history"]["objective_value_mean"])
    stds = np.array(summary["aggregated_history"]["objective_value_std"])

    plt.figure(figsize=(10, 6))
    plt.plot(evals, means, label=f'{exp_name} (mean)', color='blue')
    plt.fill_between(evals, means - stds, means + stds, color='blue', alpha=0.2, label=r'$\pm 1$ std')

    # Many optimization landscapes go to zero, log scale helps.
    # Replace zeros or negatives to avoid log errors if any exist.
    means_safe = np.where(means <= 0, 1e-10, means)
    
    plt.yscale('log')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Objective Value (Log Scale)')
    plt.title(f'Performance: {exp_name} ({summary["num_seeds"]} seeds)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    figures_dir = "results/figures"
    os.makedirs(figures_dir, exist_ok=True)
    out_file = os.path.join(figures_dir, f"{exp_name}_plot.png")
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()