
from typing import Callable, Tuple

from torch import Tensor
import torch


class DQNWrapper(torch.nn.Module):

    @staticmethod
    def _def_converter(state: object) -> Tuple[Tensor]:
        return state,

    def __init__(self,
                 nn: torch.nn.Module,
                 state_converter: Callable[[object], Tuple] = _def_converter):
        """
        Wraps a standard-nn so a conversion step can be added before forwarding
        input to nn. Expected forward input is a replay buffer state.
        :param nn:
        :param state_converter: default -> state as tuple
        """
        super().__init__()
        self._converter = state_converter
        self._nn = nn

    def forward(self, x):
        x = self._converter(x)
        return self._nn.forward(*x)
