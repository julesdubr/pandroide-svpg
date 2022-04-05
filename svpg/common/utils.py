import numpy as np
import torch as th


def obs_as_tensor(obs, device):
    """
    Moves the observation to the given device.
    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs).to(device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
