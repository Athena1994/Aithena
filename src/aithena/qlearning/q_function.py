
from abc import abstractmethod

import numpy as np
import torch

from aithena.qlearning.markov.state import DictState


class QFunction:

    @abstractmethod
    def get_q_values(self, state: DictState) -> np.ndarray:
        pass


class DeepQFunction(QFunction):

    def __init__(self, nn: torch.nn.Module):
        super().__init__()
        self._nn: torch.nn.Module = nn

    def get_q_values(self, state: DictState) -> np.ndarray:
        with torch.no_grad():
            result = self._nn(state).cpu()
        return np.array(result)
