
from dataclasses import dataclass
from typing import Set, Tuple, Union

import numpy as np
import torch

from aithena.qlearning.markov.state \
    import (StateDescriptor, TensorDictState, TensorStateBatch,
            hash_state, state_equal)


@dataclass
class Experience:
    old_state: TensorDictState
    action: int
    reward: float
    new_state: TensorDictState
    terminal: bool = False

    def __hash__(self):
        """Returns a hash of the Experience object."""
        return hash((hash_state(self.old_state), self.action,
                     self.reward, hash_state(self.new_state), self.terminal))

    def __eq__(self, other):
        """Checks if two Experience objects are equal."""
        if not isinstance(other, Experience):
            return False
        return (state_equal(self.old_state, other.old_state) and
                self.action == other.action and
                self.reward == other.reward and
                state_equal(self.new_state, other.new_state) and
                self.terminal == other.terminal)


class ExperienceBatch:

    def __init__(self,
                 old_state: TensorStateBatch,
                 action: torch.Tensor,
                 reward: torch.Tensor,
                 new_state: TensorStateBatch,
                 terminal: np.ndarray,
                 state_descriptor: StateDescriptor = None):

        self.old_states = old_state
        self.actions = action
        self.rewards = reward
        self.new_states = new_state
        self.terminal = terminal

        if state_descriptor is None:
            self._state_descriptor \
                = {k: v.shape[1:] for k, v in self.old_states.items()}
        else:
            self._state_descriptor = state_descriptor

        buffer_shape = list(old_state.values())[0].shape

        state_dim_cnt = len(list(self._state_descriptor.values())[0])

        self._size = buffer_shape[:len(buffer_shape) - state_dim_cnt]

    @staticmethod
    def create_empty(size: Union[int, Tuple[int, ...]],
                     state_descriptor: dict, cuda: bool,
                     dtype: torch.dtype = torch.float32) -> 'ExperienceBatch':
        """Creates an empty ExperienceBatch with the given size and state
        descriptor."""
        if isinstance(size, int):
            size = (size,)

        device = 'cuda' if cuda else 'cpu'

        old_state = {k: torch.zeros(size + v, dtype=dtype, device=device)
                     for k, v in state_descriptor.items()}
        new_state = {k: torch.zeros(size + v, dtype=dtype, device=device)
                     for k, v in state_descriptor.items()}
        return ExperienceBatch(
            old_state=old_state,
            action=torch.zeros(size, dtype=torch.long, device=device),
            reward=torch.zeros(size, dtype=dtype, device=device),
            new_state=new_state,
            state_descriptor=state_descriptor,
            terminal=torch.zeros(size, dtype=torch.bool, device=device)
        )

    # --- properties ---

    # def as_tensor_batch(self, cuda: bool,
    #                     dtype_: torch.dtype = torch.float32) -> TensorBatch:
    #     """Returns the ExperienceBatch as a dictionary of torch tensors."""
    #     device = 'cuda' if cuda else 'cpu'

    #     return TensorBatch(
    #         old_states=arraydict_to_tensordict(self._old_state, cuda, dtype_),
    #         actions=torch.tensor(self._action, dtype=torch.long, device=
    # device),
    #         rewards=torch.tensor(self._reward, dtype=dtype_, device=device),
    #         new_states=arraydict_to_tensordict(self._new_state, cuda, dtype_),
    #         terminal=torch.tensor(self._terminal, dtype=torch.bool,
    #                               device=device)
    #     )

    @property
    def size(self) -> Tuple[int, ...]:
        """Returns the number of experiences in the batch."""
        return self._size

    @property
    def state_descriptor(self) -> StateDescriptor:
        """Returns the state descriptor of the batch."""
        return self._state_descriptor

    @property
    def state_keys(self) -> Set[str]:
        """Returns the state descriptor of the batch."""
        return set(self.old_states.keys())

    @property
    def experiences(self) -> list[Experience]:
        """Returns the experiences in the batch as a list of Experience
        objects."""
        flat_batch = self.flatten() if len(self.size) > 1 else self
        return [flat_batch[i] for i in range(len(flat_batch))]

    # -- built-in methods ---

    def __getitem__(self, ix) -> Union[Experience, 'ExperienceBatch']:
        """Returns the Experience at the given index."""
        if not isinstance(ix, tuple):
            ix = (ix,)

        ix_dim = len(ix)
        size_dim = len(self.size)

        if ix_dim > size_dim:
            raise IndexError(f'Index {ix} does not match batch size '
                             f'{self.size}')

        if any(isinstance(i, slice) for i in ix) or ix_dim < size_dim:
            return self._slice(ix)

        return self._get_item(ix)

    def __setitem__(self, ix, value: Experience):
        """Sets the Experience at the given index."""
        if not isinstance(ix, tuple):
            ix = (ix,)
        if len(ix) != len(self.size):
            raise IndexError(f'Index {ix} does not match batch size '
                             f'{self.size}')

        for k in self.state_keys:
            self.old_states[k][*ix] = value.old_state[k]
            self.new_states[k][*ix] = value.new_state[k]

        self.actions[*ix] = value.action
        self.rewards[*ix] = value.reward
        self.terminal[*ix] = value.terminal

    def __len__(self) -> int:
        """Returns the number of experiences in the batch."""
        return len(self.actions)

    # --- public methods ---

    def flatten(self, dim_cnt: int = -1) -> 'ExperienceBatch':
        """
        Flattens the first dim_cnt dimenstions ExperienceBatch.
        If dim_cnt is -1, output size is 1D.
        """

        if len(self._size) == 1:
            return self

        if dim_cnt == -1:
            return self.flatten(len(self._size))

        if dim_cnt < 1 or dim_cnt > len(self._size):
            raise ValueError(f'dim_cnt must be between 1 and {len(self._size)}')

        new_size = (np.prod(self._size[:dim_cnt]), *self._size[dim_cnt:])

        flat_old_state = {k: v.reshape((-1, *v.shape[dim_cnt:]))
                          for k, v in self.old_states.items()}
        flat_new_state = {k: v.reshape((-1, *v.shape[dim_cnt:]))
                          for k, v in self.new_states.items()}

        return ExperienceBatch(
            old_state=flat_old_state,
            action=self.actions.reshape(new_size),
            reward=self.rewards.reshape(new_size),
            new_state=flat_new_state,
            terminal=self.terminal.reshape(new_size),
            state_descriptor=self.state_descriptor
        )

    def sample(self, ix_array: np.ndarray) -> 'ExperienceBatch':
        """Returns a sampled ExperienceBatch from the given index array."""
        return ExperienceBatch(
            old_state={k: v[ix_array] for k, v in self.old_states.items()},
            action=self.actions[ix_array],
            reward=self.rewards[ix_array],
            new_state={k: v[ix_array] for k, v in self.new_states.items()},
            terminal=self.terminal[ix_array],
            state_descriptor=self._state_descriptor
        )

    # --- private methods ---
    def _get_item(self, ix: Tuple[int]):
        if len(ix) != len(self.size):
            raise IndexError(f'Index {ix} does not match batch size '
                             f'{self.size}')

        return Experience(
            old_state={k: v[*ix, ...] for k, v in self.old_states.items()},
            action=self.actions[*ix],
            reward=self.rewards[*ix],
            new_state={k: v[*ix, ...] for k, v in self.new_states.items()},
            terminal=self.terminal[*ix],
        )

    def _slice(self, ix: Tuple[Union[int, slice]])\
            -> 'ExperienceBatch':
        if len(ix) > len(self.size):
            raise IndexError(f'Index {ix} does not match batch size '
                             f'{self.size}')

        """Returns a sliced ExperienceBatch from start to end."""
        return ExperienceBatch(
            old_state={k: v[*ix] for k, v in self.old_states.items()},
            action=self.actions[*ix],
            reward=self.rewards[*ix],
            new_state={k: v[*ix] for k, v in self.new_states.items()},
            terminal=self.terminal[*ix],
            state_descriptor=self._state_descriptor
        )
