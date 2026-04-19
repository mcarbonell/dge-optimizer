import platform
import subprocess
import os
import json
from datetime import datetime

def get_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "unknown"

def get_system_info():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }

def setup_result_directories():
    base_dir = "results"
    dirs = [
        os.path.join(base_dir, "raw"),
        os.path.join(base_dir, "summary"),
        os.path.join(base_dir, "figures")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def save_raw_result(experiment_name, seed, config, history, metrics):
    setup_result_directories()
    timestamp = datetime.now().isoformat()
    
    data = {
        "experiment_name": experiment_name,
        "seed": seed,
        "timestamp": timestamp,
        "commit_hash": get_commit_hash(),
        "hardware": get_system_info(),
        "config": config,
        "metrics": metrics,
        "history": history
    }
    
    filename = f"results/raw/{experiment_name}_seed{seed}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    return filename