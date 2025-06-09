
from typing import Callable, Iterable, List, Tuple
import numpy as np
import torch


from aithena.qlearning.markov.experience import Experience, ExperienceBatch
from aithena.qlearning.markov.state import StateDescriptor


class ReplayBuffer:

    def __init__(self, capacity: int, desc: StateDescriptor,
                 cuda: bool = False, dtype: torch.dtype = torch.float32):
        self._buffer = ExperienceBatch.create_empty(capacity, desc,
                                                    cuda, dtype)
        self._shapes = desc

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
    def state_descriptor(self) -> StateDescriptor:
        '''Returns the state descriptor of the replay buffer.'''
        return self._shapes

    # --- buildin methods ---

    def __len__(self):
        return self._cnt

    # --- properties ---
    @property
    def buffer(self) -> ExperienceBatch:
        return self._buffer[:self.size]

    @property
    def experiences(self) -> List[Experience]:
        return self.buffer.experiences

    # --- public methods ---

    def clear(self):
        self._cnt = 0
        self._weight_sum = 0
        self._weights[:] = 0

    def add_experiences(self, experiences: Iterable[Experience],
                        weight_callback: Callable[[Experience], float]):
        for exp in experiences:
            self.add_experience(exp, weight_callback(exp))

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

        if experience.new_state is None:
            experience.new_state = {k: torch.zeros_like(v)
                                    for k, v in experience.old_state.items()}

        self._buffer[next_ix] = experience

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

        return self._buffer.sample(choices)
