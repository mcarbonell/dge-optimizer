from .optimizer import DGEOptimizer

try:
    from .torch_optimizer import TorchDGEOptimizer
    __all__ = ["DGEOptimizer", "TorchDGEOptimizer"]
except ImportError:
    __all__ = ["DGEOptimizer"]
