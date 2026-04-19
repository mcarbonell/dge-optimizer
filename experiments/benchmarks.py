import numpy as np

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    return 10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))

def ackley(x):
    d = len(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def ellipsoid(x):
    d = len(x)
    indices = np.arange(1, d + 1)
    weights = 1000.0 ** ((indices - 1) / (d - 1)) if d > 1 else np.array([1.0])
    return np.sum(weights * x**2)

def sparse_sphere(x):
    # Only top 10% of variables contribute to the loss
    d = len(x)
    k = max(1, d // 10)
    return np.sum(x[:k]**2)

def step_sphere(x):
    # Non-differentiable: piecewise flat
    return np.sum(np.floor(x)**2)

BENCHMARKS = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "ellipsoid": ellipsoid,
    "sparse_sphere": sparse_sphere,
    "step_sphere": step_sphere
}

def get_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name}")
    return BENCHMARKS[name]
