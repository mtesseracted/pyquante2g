# helper functions for getting gpu information

def get_gpu0_torch():
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
    except Exception:
        return None


def get_gpu0_tensorflow():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU") # TF2 style
        if gpus:
            # dev = gpus[0]
            # devices["tensorflow"] = dev.name
            return "/GPU:0"
    except Exception:
        return None


def get_gpu0_cupy():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cuda:0"
    except Exception:
        return None

def get_gpu0_jax():
    try:
        import jax
        gpus = jax.devices("gpu")
        if gpus:
            d0 = gpus[0]
            # devices["jax"] = f"{d0.platform}:{d0.id} ({d0.device_kind})"
            return f"{d0.platform}:{d0.id}"
    except Exception:
        return None


def get_gpu0(backend):

    if backend == "jax":
        return get_gpu0_jax()

    elif backend == "cupy":
        return get_gpu0_cupy()

    if backend == "torch":
        return get_gpu0_torch()

    elif backend == "tensorflow":
        return get_gpu0_tensorflow()

    else:
        raise ValueError(f"Uknown backend type {backend}")



def detect_gpu0_devices(found_modules):
    """
    Returns a dict mapping backend name -> GPU string for the first GPU it sees.
    """
    devices = {backend: get_gpu0(backend) for backend in found_modules}
    return devices


def get_array_device_label(x):
    dev = getattr(x, "device", None)
    if callable(dev):
        dev = dev()
    if dev is None:
        devices = getattr(x, "devices", None)
        if callable(devices):
            devices = list(devices())
            if len(devices) == 1:
                dev = devices[0]
    if dev is None:
        return None
    if hasattr(dev, "platform") and hasattr(dev, "id"):
        return f"{dev.platform}:{dev.id}"
    if hasattr(dev, "type"):
        if getattr(dev, "index", None) is None:
            return dev.type
        return f"{dev.type}:{dev.index}"
    if hasattr(dev, "id"):
        return f"cuda:{int(dev.id)}"
    return str(dev)
