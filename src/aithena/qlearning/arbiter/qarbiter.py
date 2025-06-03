from abc import abstractmethod

import numpy as np

from aithena.qlearning.markov.state import DictState
from aithena.qlearning.q_function import QFunction


class QArbiter:
    def __init__(self,
                 q_fct: QFunction):
        self._q_fct = q_fct

    def decide(self, state: DictState) -> np.ndarray:
        q_values = self._q_fct.get_q_values(state)
        if len(q_values.shape) == 1:
            q_values = q_values.reshape(1, -1)

        return self.make_decision(q_values)

    @abstractmethod
    def make_decision(self, q_values: np.ndarray) -> np.array:
        pass
