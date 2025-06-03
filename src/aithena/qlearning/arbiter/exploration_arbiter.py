
from abc import abstractmethod

import numpy as np
from torch.nn import Module

from aithena.qlearning.arbiter.qarbiter import QArbiter
from aithena.qlearning.policy import QDecisionPolicy
from aithena.qlearning.q_function import DeepQFunction, QFunction
from jodisutils.config.decorator import config


@config
class ExplorationArbiterConfig:
    type: str
    params: dict

    def create_arbiter(self, q_fct: Module | QFunction) -> 'ExplorationArbiter':
        """
        Create an exploration arbiter based on the type and parameters provided.
        :param nn: The neural network module to be used with the QFunction.
        :return: An instance of the exploration arbiter.
        """
        if isinstance(q_fct, Module):
            q_fct = DeepQFunction(q_fct)

        t = self.type.lower()
        if t == 'epsilon-greedy':
            from aithena.qlearning.arbiter.epsilon_greedy_arbiter \
                import EpsilonGreedyConfig
            return EpsilonGreedyConfig(**self.params).create(q_fct)
        elif t == 'decaying-epsilon-greedy':
            from aithena.qlearning.arbiter.decaying_epsilon_greedy_arbiter \
                import DecayingEpsilonGreedyConfig
            return DecayingEpsilonGreedyConfig(**self.params).create(q_fct)
        else:
            raise ValueError(f"Unknown exploration arbiter type: {self.type}")


class ExplorationContext:
    def __init__(self, arbiter: 'ExplorationArbiter', val: bool):
        self._arbiter = arbiter
        self._old_val = arbiter.explore
        self._val = val

    def __enter__(self):
        self._arbiter.explore = self._val

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._arbiter.explore = self._old_val


class ExplorationArbiter(QArbiter):

    def __init__(self,
                 q_fct: QFunction,
                 decision_policy: QDecisionPolicy):
        super().__init__(q_fct)
        self._explore = False
        self._decision_policy = decision_policy

        self._exploration_index = 0  # number of times the arbiter has been used

    # --- abstract methods ---

    @abstractmethod
    def explore_actions(self,
                        policy_actions: np.array,
                        q_values: np.array) -> np.array:
        pass

    # --- properties ---

    @property
    def explore(self) -> bool:
        return self._explore

    @explore.setter
    def explore(self, value: bool):
        self._explore = value

    @property
    def exploration_index(self) -> int:
        return self._exploration_index

    @exploration_index.setter
    def exploration_index(self, index: int):
        self._exploration_index = index
        self.on_exploration_index_changed(self._exploration_index)

    # --- public methods ---

    def exploration(self, val: bool = True) -> ExplorationContext:
        return ExplorationContext(self, val)

    def on_exploration_index_changed(self, ix: int):
        pass

    def make_decision(self, q_values: np.ndarray) -> np.array:
        actions = self._decision_policy.make_decision(q_values)
        if self._explore:
            actions = self.explore_actions(actions, q_values)
            self.exploration_index += 1

        return actions
