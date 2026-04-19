import numpy as np
import time
import sys
import os
import math

# Asegurar importación de V1 y V2
sys.path.append(os.path.join(os.getcwd(), 'dge-optimizer'))
from dge.optimizer import DGEOptimizer as DGE_V1
from scratch.dge_v2_optimized import DGEOptimizerV2 as DGE_V2

def sphere(x): return np.sum(x**2)

def ackley(x):
    arg1 = -0.2 * np.sqrt(np.mean(x**2))
    arg2 = np.mean(np.cos(2.0 * np.pi * x))
    return -20.0 * np.exp(arg1) - np.exp(arg2) + 20.0 + np.e

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

def run_experiment(name, opt_class, func, dim, steps, seed=42):
    rng = np.random.default_rng(seed)
    # Inicialización aleatoria centrada pero dispersa
    x = rng.standard_normal(dim).astype(np.float32) * 2.0
    
    # Parámetros consistentes
    optimizer = opt_class(dim=dim, total_steps=steps, seed=seed, lr=0.5, clip_norm=10.0)
    
    start_wall = time.perf_counter()
    eval_time = 0
    
    for _ in range(steps):
        def wrapped_f(params):
            nonlocal eval_time
            t0 = time.perf_counter()
            res = func(params)
            eval_time += (time.perf_counter() - t0)
            return res
            
        x, _ = optimizer.step(wrapped_f, x)
        
    end_wall = time.perf_counter()
    return {
        "loss": func(x),
        "wall": end_wall - start_wall,
        "overhead": (end_wall - start_wall) - eval_time
    }

if __name__ == "__main__":
    test_cases = [
        ("Sphere", sphere, [1000, 5000]),
        ("Ackley", ackley, [1000, 5000]),
        ("Rosenbrock", rosenbrock, [1000]) # Rosenbrock es extremadamente lenta en alta dim
    ]
    
    steps = 500
    
    print(f"{'Función':<12} | {'Dim':<6} | {'Algo':<8} | {'Final Loss':<12} | {'Overhead(s)':<12} | {'Speedup'}")
    print("-" * 85)
    
    for func_name, func, dims in test_cases:
        for dim in dims:
            results = {}
            for name, cls in [("V1", DGE_V1), ("V2", DGE_V2)]:
                res = run_experiment(name, cls, func, dim, steps)
                results[name] = res
            
            v1 = results["V1"]
            v2 = results["V2"]
            speedup = v1['overhead'] / v2['overhead'] if v2['overhead'] > 0 else 0
            
            print(f"{func_name:<12} | {dim:<6} | {'V1':<8} | {v1['loss']:<12.4f} | {v1['overhead']:<12.4f} | ---")
            print(f"{'':<12} | {'':<6} | {'V2':<8} | {v2['loss']:<12.4f} | {v2['overhead']:<12.4f} | {speedup:.2f}x")
            print("-" * 85)
