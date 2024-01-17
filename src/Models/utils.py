from typing import Tuple, Union

import numpy as np

import torch

def fan_out_normal_seed(
    shape: Tuple,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Randomly generates a tensor based on a seed and normal initialization
    with std 1/fan_out.
    Args:
        shape (Tuple):
            Desired shape of the tensor.
        device (torch.device or str, optional):
            Device to generate tensor on. Defaults to "cuda".
        seed (int, optional):
            The seed. Defaults to 42.
        dtype (torch.dtype, optional):
            Tensor type. Defaults to torch.float32.
    Returns:
        torch.Tensor: The randomly generated tensor
    """
    torch.manual_seed(seed)
    a = torch.zeros(shape, dtype=dtype)
    torch.nn.init.normal_(a, std=1 / shape[1])
    return a

def batcher(batch_size, kwargs):
    batches = []
    keys = list(kwargs.keys())
    n = len(kwargs[keys[0]])
    for s in np.arange(0, n, batch_size):
        batch = {}
        for key in keys:
            batch[key] = kwargs[key][s:s + batch_size]
        batches.append(batch)
    return batches

def generate_rotation(r, theta):
    """Generate a rotation from polar representation.

    Parameters
    ----------
    r: vector (N x 1)
        rotation axis
    theta: float
        Rotation angle

    Returns
    -------
    Cayley transformed rotation matrix

    Notes
    -----
    https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix
    """
    I = np.eye(r.shape[0])
    S = theta * (r@r.T - r.T@r)
    return np.linalg.inv(I + S)@(I - S)


###TODO: keep for storage atm
def label_processor(labels):
    return torch.tensor([int(k) for label in labels.values() for k in label])

def flatten_output(output, labels=None):
    out = torch.cat([l for t in output.values() for l in t], dim=0)
    if labels is not None:
        return out, label_processor(labels)
    else:
        return out