
import numpy as np
from aithena.qlearning.arbiter.epsilon_greedy_arbiter\
      import EpsilonGreedyArbiter
from aithena.qlearning.q_function import QFunction
from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config


@config
class DecayingEpsilonGreedyConfig:
    epsilon_start: float = Attribute('epsilon-start')
    epsilon_end: float = Attribute('epsilon-end')
    decay: float = Attribute('decay')

    def create(self, q_fct: QFunction) -> 'DecayingEpsilonGreedyArbiter':
        return DecayingEpsilonGreedyArbiter(
            q_fct, self.epsilon_start, self.epsilon_end, self.decay)


class DecayingEpsilonGreedyArbiter(EpsilonGreedyArbiter):

    def __init__(self,
                 q_fct: QFunction,
                 epsilon_start: float,
                 epsilon_end: float,
                 decay: float):
        super().__init__(q_fct, epsilon_start)

        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._decay = decay

    def on_exploration_index_changed(self, ix: int):

        self._epsilon = self._epsilon_end \
            + (self._epsilon_start - self._epsilon_end) * \
            np.exp(-1 * ix / self._decay)
