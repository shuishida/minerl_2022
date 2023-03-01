import warnings
from typing import Dict, Tuple, Union, Callable, Any, List

import gym
import numpy as np
import torch
from gym import spaces
from torch.distributions import Categorical
from torch.nn import functional as F

from src.utils.image_utils import torch_to_image, image_to_torch

"""
Partly borrowed from
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py
"""


def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).

    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).

    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0


def is_image_space(
        observation_space: spaces.Space,
        check_channels: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.

    Valid images: RGB, RGBD, GrayScale with values in [0, 255]

    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :return:
    """
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # RGB, RGBD, GrayScale
        return n_channels in [1, 3, 4]
    return False


def transpose_image_space(observation_space: spaces.Box) -> spaces.Box:
    """
    Transpose an observation space (re-order channels).

    :param observation_space:
    :return:
    """
    height, width, channels = observation_space.shape
    new_shape = (channels, height, width)
    return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)


class OneHotEncoding(spaces.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(n=4)
    """

    def __init__(self, n=None):
        assert isinstance(n, int) and n > 0
        self.n = n
        spaces.Space.__init__(self, (n,), np.float)

    def sample(self):
        one_hot_vector = torch.zeros(self.n, dtype=torch.float)
        one_hot_vector[np.random.randint(self.n)] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray, torch.Tensor)):
            number_of_zeros = np.sum([e == 0 for e in x])
            number_of_ones = np.sum([e == 1 for e in x])
            return (number_of_zeros == (self.n - 1)) and (number_of_ones == 1)
        else:
            return False

    def __repr__(self):
        return f"OneHotEncoding{self.n}"

    def __eq__(self, other):
        return self.n == other.n


def transform_space(space: spaces.Space, image_transforms=None) -> Tuple[spaces.Space, Callable[[Any], Any], Callable[[Any, bool], Any]]:
    """
    :param space:
    :param image_transforms:
    :return: space, transform, inverse_transforms
    """
    if not image_transforms:
        image_transforms = lambda x: x

    if isinstance(space, spaces.Box):
        if is_image_space(space):
            if is_image_space_channels_first(space):
                return space, image_transforms, lambda x, _: torch_to_image(x, transpose=False)
            else:
                return transpose_image_space(space), lambda x: image_transforms(image_to_torch(x)), lambda x, _: torch_to_image(x)
        return space, lambda x: x, lambda x, _: x

    elif isinstance(space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return OneHotEncoding(space.n), \
               lambda x: F.one_hot(torch.as_tensor(x, dtype=torch.long), num_classes=space.n).float(), \
               lambda x, deterministic: (x.argmax(dim=-1) if deterministic else Categorical(torch.softmax(x, dim=-1)).sample()).cpu().data.numpy()

    elif isinstance(space, spaces.MultiDiscrete):
        if not len(space.shape) == 1:
            raise NotImplementedError
        return transform_space(gym.spaces.Tuple(OneHotEncoding(int(n)) for n in space.nvec), image_transforms)

    elif isinstance(space, spaces.MultiBinary):
        return space, lambda x: x.float(), lambda x, deterministic: ((x >= 0).float() if deterministic else torch.bernoulli(torch.sigmoid(x))).cpu().data.numpy()

    elif isinstance(space, spaces.Tuple):
        trans_spaces = []
        trans_funcs: List[Callable[[Any], Any]] = []
        inv_funcs: List[Callable[[Any, bool], Any]] = []
        for s in space:
            s, f, inv = transform_space(s, image_transforms)
            trans_spaces.append(s)
            trans_funcs.append(f)
            inv_funcs.append(inv)
        return spaces.Tuple(*trans_spaces), \
               lambda x: (func(x) for func in trans_funcs), \
               lambda x, deterministic: (inv(x, deterministic) for inv in inv_funcs)

    elif isinstance(space, spaces.Dict):
        trans_spaces = spaces.Dict()
        trans_funcs: Dict[Callable[[Any], Any]] = {}
        inv_funcs: Dict[Callable[[Any, bool], Any]] = {}
        for k in space:
            trans_spaces.spaces[k], trans_funcs[k], inv_funcs[k] = transform_space(space[k], image_transforms)
        return trans_spaces, \
               lambda x: {k: func(x[k]) for k, func in trans_funcs.items()}, \
               lambda x, deterministic: {k: inv(x[k], deterministic) for k, inv in inv_funcs.items()}

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {space}")

# --------------------------
# Some unused util functions
# --------------------------


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def check_for_nested_spaces(obs_space: spaces.Space):
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.

    :param obs_space: an observation space
    :return:
    """
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = obs_space.spaces.values() if isinstance(obs_space, spaces.Dict) else obs_space.spaces
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )
