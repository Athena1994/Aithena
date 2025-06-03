from typing import Tuple, Union, TypeAlias

import numpy as np
import torch


StateDescriptor: TypeAlias = dict[str, Union[int, Tuple[int, ...]]]
DictState: TypeAlias = dict[str, np.ndarray]
TensorDictState: TypeAlias = dict[str, torch.Tensor]
StateBatch: TypeAlias = dict[str, np.ndarray]
TensorStateBatch: TypeAlias = dict[str, torch.Tensor]


def arraydict_to_tensordict(
        array_dict: dict[str, np.ndarray],
        cuda: bool = False,
        dtype_: torch.dtype = torch.float32)\
        -> dict[str, torch.Tensor]:
    """
    Converts a DictState to a TensorDictState.
    """
    device = 'cuda' if cuda else 'cpu'
    return {field: torch.tensor(value, dtype=dtype_, device=device)
            for field, value in array_dict.items()}


def move_tensor_state(tensor_state: TensorDictState, cuda: bool) \
        -> TensorDictState:
    """
    Moves a TensorDictState to the specified device (CPU or CUDA).
    """
    device = 'cuda' if cuda else 'cpu'
    return {field: value.to(device) for field, value in tensor_state.items()}


def create_empty_state(desc: StateDescriptor) -> DictState:
    """
    Creates an empty Markov state with the given descriptor and size.
    (mostly for testing purposes)
    """
    return {field: np.zeros(shape) for field, shape in desc.items()}


def create_random_state(desc: StateDescriptor) -> TensorDictState:
    """
    Creates an random Markov state with the given descriptor and size.
    (mostly for testing purposes)
    """
    return {field: torch.Tensor(np.random.uniform(size=shape))
            for field, shape in desc.items()}


def make_descriptor(state: DictState) -> StateDescriptor:
    """Creates a state descriptor from a Markov state."""
    return {field: desc.shape for field, desc in state.items()}


def state_equal(state1: DictState, state2: DictState) -> bool:
    """Checks if two Markov states are equal."""

    desc = make_descriptor(state1)

    if desc != make_descriptor(state2):
        return False

    return not any({not np.array_equal(state1[k], state2[k]) for k in desc})


def hash_state(state: DictState) -> int:
    """Returns a hash of the Markov state."""
    return hash(tuple((k, tuple(v.flatten())) for k, v in state.items()))
