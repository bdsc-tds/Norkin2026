import os
import random
import numpy as np
import importlib.util


def set_all_seeds(seed=0):
    """
    Sets seeds and environment variables for reproducibility.
    Checks for library availability using importlib.
    """

    # 1. Environment Variables (Must be set before libraries are loaded)
    # ----------------------------------------------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["NUMBA_CPU_NAME"] = "generic"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 2. Core Python and Numpy
    # ------------------------
    random.seed(seed)
    np.random.seed(seed)

    # 3. Conditional Seeding with importlib
    # -------------------------------------

    # PyTorch
    if importlib.util.find_spec("torch"):
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set to True/False based on whether you want code to crash on non-deterministic ops
        torch.use_deterministic_algorithms(True, warn_only=True)

    # CuPy
    if importlib.util.find_spec("cupy"):
        import cupy

        cupy.random.seed(seed)

    # TensorFlow
    if importlib.util.find_spec("tensorflow"):
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Force TF to be deterministic on GPU
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # JAX
    if importlib.util.find_spec("jax"):
        # Note: JAX reproducibility is handled via explicit PRNGKeys in code,
        # but environment variables help control XLA backend behavior.
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"

    print(f"Seed: {seed}")


def get_seed_worker(seed=0):
    """
    Returns a worker_init_fn for PyTorch DataLoaders.
    Usage: DataLoader(..., worker_init_fn=get_seed_worker(42))
    """

    def seed_worker(worker_id):
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker
