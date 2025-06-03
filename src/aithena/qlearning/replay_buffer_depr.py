
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import numpy as np


from aithena.qlearning.markov.experience import Experience
from aithena.qlearning.markov.state import StateDescriptor


@dataclass
class ExperienceBatch:
    prev_state: Union[Dict[str, np.ndarray], np.ndarray]
    action: np.ndarray
    reward: np.ndarray
    next_state: Union[Dict[str, np.ndarray], np.ndarray]


class ReplayBuffer:

    def __init__(self, capacity: int, state: StateDescriptor = None):
        self._reward_buffer: np.ndarray = np.array([.0] * capacity)
        self._action_buffer: np.ndarray = np.array([-1] * capacity)
        self._prev_state_buffer: Dict[str, object] = {}
        self._next_state_buffer: Dict[str, object] = {}

        if state is None:
            state = StateDescriptor({None: StateDescriptor.Field((1,))})
        self._state_descriptor = state

        for field, desc in state.fields.items():
            if desc.tensor:
                self._prev_state_buffer[field] = np.zeros((capacity,
                                                           *desc.shape))
                self._next_state_buffer[field] = np.zeros((capacity,
                                                           *desc.shape))
            else:
                self._prev_state_buffer[field] = np.array([None] * capacity)
                self._next_state_buffer[field] = np.array([None] * capacity)

        self._cnt = 0
        self._capacity = capacity

        self._weights = np.zeros(capacity)
        self._weight_sum = 0  # keep track of new/removed rewards for probs

    @property
    def is_empty(self) -> bool:
        '''Returns True if the replay buffer is empty, otherwise False.'''
        return self._cnt == 0

    @property
    def capacity(self) -> int:
        '''Returns the maximum number of experiences that can be stored in the
        replay buffer.'''
        return self._capacity

    @property
    def size(self) -> int:
        '''Returns the current number of experiences stored in the replay
        buffer.'''
        return self._cnt

    @property
    def is_full(self) -> bool:
        '''Returns True if the replay buffer is full, otherwise False.'''
        return self._cnt == self._capacity

    @property
    def object_states(self) -> bool:
        return len(self._state_descriptor.fields) == 1 \
            and None in self._state_descriptor.fields

    # --- buildin methods ---

    def __len__(self):
        return self._cnt

    # --- properties ---
    @property
    def all(self) -> List[Experience]:
        '''Returns all stored experiences in the buffer as a list of Experience
        objects.'''

        def invert(d: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
            return [{k: d[k][i] for k in d} for i in range(self.capacity)]

        if self.object_states:
            return [Experience(*t) for t in list(zip(
                self._prev_state_buffer[None],
                self._action_buffer,
                self._reward_buffer,
                self._next_state_buffer[None]))[:self._cnt]]
        else:
            return [Experience(*t) for t in list(zip(
                invert(self._prev_state_buffer),
                self._action_buffer,
                self._reward_buffer,
                invert(self._next_state_buffer)))[:self._cnt]]

    # --- public methods ---

    def clear(self):
        self._cnt = 0
        self._weight_sum = 0
        self._weights[:] = 0

    def add_experience(self, experience: Experience, weight: float):
        if not isinstance(experience, Experience):
            raise Exception("parameter must be of type Experience")
        if weight <= 0:
            raise Exception(f'Experience weight must not be <= 0 ({weight})')

        if self.is_full:
            next_ix = np.random.randint(0, self._cnt)
        else:
            next_ix = self._cnt

        # keep track of cumulated weights for sample probability
        self._weight_sum += weight - self._weights[next_ix]
        self._weights[next_ix] = weight

        # wrap experience in dict if object state
        if self.object_states:
            old_state = {None: experience.old_state}
            new_state = {None: experience.new_state}
        else:
            descriptor = self._state_descriptor.fields
            old_state = {k: np.reshape(v, descriptor[k].shape)
                         for k, v in experience.old_state.items()}
            new_state = {k: np.reshape(v, descriptor[k].shape)
                         for k, v in experience.new_state.items()}

        for field in old_state:
            self._prev_state_buffer[field][next_ix, ...] = old_state[field]
            self._next_state_buffer[field][next_ix, ...] = new_state[field]

        self._action_buffer[next_ix] = experience.action
        self._reward_buffer[next_ix] = experience.reward

        # cycle buffer pointer
        if self._cnt != self._capacity:
            self._cnt += 1

    def sample_experiences(self, size: int | Tuple, replace: bool)\
            -> ExperienceBatch:
        # maybe include priority selection for buy/sell experiences?

        cnt = size[-1] if isinstance(size, Tuple) else size

        if self.is_empty:
            raise Exception("can not sample on empty replay buffer")
        if self.size < cnt and not replace:
            raise Exception("not enough samples in replay buffer")

        choices = np.random.choice(
            range(self.size), size,
            replace=replace,
            p=self._weights[:self.size] / self._weight_sum)

        if self.object_states:
            return ExperienceBatch(
                prev_state=self._prev_state_buffer[None][choices],
                action=self._action_buffer[choices],
                reward=self._reward_buffer[choices],
                next_state=self._next_state_buffer[None][choices],
            )
        else:
            return ExperienceBatch(
                prev_state={k: self._prev_state_buffer[k][choices]
                            for k in self._prev_state_buffer},
                action=self._action_buffer[choices],
                reward=self._reward_buffer[choices],
                next_state={k: self._next_state_buffer[k][choices]
                            for k in self._next_state_buffer},
            )
