from collections import OrderedDict

import numpy as np
import torch
from einops import rearrange
from torch import Tensor


def apply_func_dict(data, func):
    if isinstance(data, OrderedDict):
        return OrderedDict(
            [(k, apply_func_dict(v, func)) for k, v in data.items()]
        )
    elif isinstance(data, dict):
        return {k: apply_func_dict(v, func) for k, v in data.items()}
    else:
        return func(data)


def apply_func(data, func):
    if isinstance(data, OrderedDict):
        return OrderedDict(
            [(k, apply_func(v, func)) for k, v in data.items()]
        )
    elif isinstance(data, dict):
        return {k: apply_func(v, func) for k, v in data.items()}
    elif isinstance(data, list):
        return [apply_func(e, func) for e in data]
    elif isinstance(data, tuple):
        return (apply_func(e, func) for e in data)
    else:
        return func(data)


def to_torch(data):
    def func(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, (float, int, bool)):
            return torch.as_tensor(x)
        else:
            return x
    return apply_func(data, func)


def to_numpy(data):
    return apply_func(data, lambda x: x.cpu().data.numpy() if isinstance(x, torch.Tensor) else x)


def to_cpu(data):
    return apply_func(data, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)


def to_item(data):
    return apply_func(data, lambda x: x.item() if isinstance(x, torch.Tensor) else x)


def to_device(data, device):
    return apply_func(data, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)


def nested_slice(data, slc):
    return apply_func_dict(data, lambda x: x[slc])


def nested_shape(data):
    return apply_func(data, lambda x: x.shape if isinstance(x, (torch.Tensor, np.ndarray)) else x)


def nested_collate(tree, batch = None):
    if isinstance(tree, OrderedDict) or isinstance(tree, dict):
        if batch:
            assert batch.keys() == tree.keys()
            for k, v in tree.items():
                batch[k] = nested_collate(v, batch[k])
            return batch
        elif isinstance(tree, OrderedDict):
            return OrderedDict(
                [(k, nested_collate(v)) for k, v in tree.items()]
            )
        else:
            return {k: nested_collate(v) for k, v in tree.items()}
    else:
        if batch:
            assert isinstance(batch, list)
            batch.append(tree)
            return batch
        return [tree]


def nested_reduce(data, reduce: str = "stack"):
    def func(x):
        try:
            if reduce == "stack":
                if isinstance(x[0], np.ndarray):
                    return np.stack(x)
                elif isinstance(x[0], torch.Tensor):
                    return torch.stack(x)
                return torch.as_tensor(x)
            elif reduce == "cat":
                if isinstance(x[0], np.ndarray):
                    return np.concatenate(x)
                elif isinstance(x[0], torch.Tensor):
                    return torch.cat(x)
                return torch.as_tensor(x)
            else:
                print(f"Reduce type {reduce} not registered")
                return x
        except Exception as e:
            return x
    return apply_func_dict(data, func)


def tree_batch_collate(batch, reduce: str = "stack"):
    results = None
    for batch_elem in batch:
        results = nested_collate(batch_elem, results)
    return nested_reduce(results, reduce)


def nested_rearrange(data, *args, **kwargs):
    return apply_func(data, lambda x: rearrange(x, *args, **kwargs) if isinstance(x, (np.ndarray, torch.Tensor)) else x)


def nested_einsum(equation, data, *operands):
    return apply_func(data, lambda x: torch.einsum(equation, x, *operands) if isinstance(x, torch.Tensor) else x)


def nested_sum(data) -> Tensor:
    if isinstance(data, dict):
        return torch.stack([nested_sum(x) for x in data.values()]).sum()
    elif isinstance(data, (list, tuple)):
        return torch.stack([nested_sum(x) for x in data]).sum()
    else:
        return data


def inverse_collate(data, batch_size):
    unpackable_datatype = (OrderedDict, dict, list, tuple, torch.Tensor, np.ndarray)
    if isinstance(data, OrderedDict):
        return [OrderedDict(
            [(k, inverse_collate(v, batch_size))[i] for k, v in data.items()]
        ) for i in range(batch_size)]
    elif isinstance(data, dict):
        return [{k: inverse_collate(v, batch_size)[i] for k, v in data.items()} for i in range(batch_size)]
    elif isinstance(data, list):
        if not isinstance(data[0], unpackable_datatype):
            assert len(data) == batch_size
            return data
        return [[inverse_collate(d, batch_size)[i] for d in data] for i in range(batch_size)]
    elif isinstance(data, tuple):
        if not isinstance(data[0], unpackable_datatype):
            assert len(data) == batch_size
            return list(data)
        return [(inverse_collate(d, batch_size)[i] for d in data) for i in range(batch_size)]
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        assert len(data) == batch_size
        return [d for d in data]
    else:
        return data


def nested_lambda(func, *args):
    if isinstance(args[0], OrderedDict):
        return OrderedDict(
            [(k, nested_lambda(func, *(arg[k] for arg in args))) for k in args[0]]
        )
    elif isinstance(args[0], dict):
        return {k: nested_lambda(func, *(arg[k] for arg in args)) for k in args[0]}
    elif isinstance(args[0], list):
        return [nested_lambda(func, *(arg[i] for arg in args)) for i in range(len(args[0]))]
    elif isinstance(args[0], tuple):
        return (nested_lambda(func, *(arg[i] for arg in args)) for i in range(len(args[0])))
    else:
        return func(*args)
