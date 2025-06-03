

import numpy as np
from aithena.qlearning.arbiter.exploration_arbiter import ExplorationArbiter
from aithena.qlearning.policy import QGreedyPolicy
from aithena.qlearning.q_function import QFunction
from jodisutils.config.decorator import config


@config
class EpsilonGreedyConfig:
    epsilon: float = 0.05

    def create(self, q_fct: QFunction) -> 'EpsilonGreedyArbiter':
        return EpsilonGreedyArbiter(q_fct, self.epsilon)


class EpsilonGreedyArbiter(ExplorationArbiter):

    def __init__(self, q_fct: QFunction, epsilon: float):
        super().__init__(q_fct, QGreedyPolicy())
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def explore_actions(self,
                        policy_actions: np.array,
                        q_values: np.array):
        randomize_choices = np.random.uniform(
            0, 1, size=policy_actions.shape) <= self._epsilon

        policy_actions[randomize_choices] \
            = np.random.choice(range(q_values.shape[-1]),
                               np.sum(randomize_choices))
        return policy_actions
