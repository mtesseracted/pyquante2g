import numpy as np

import pyquante2g.torch_ops as tops


TORCH_FLOAT_DTYPES = ("float32", "float64")


def make_torch_ops(default_dtype="float64", device=None):
    return tops.TorchOps(
        device=device,
        default_dtype=default_dtype,
        int_dtype="int64",
    )


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)
