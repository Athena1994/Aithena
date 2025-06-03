

import numpy as np


class QDecisionPolicy:
    def __init__(self):
        pass

    def make_decision(self, q_values: np.ndarray) -> np.array:
        pass


class QGreedyPolicy(QDecisionPolicy):
    def make_decision(self, q_values: np.ndarray) -> np.array:
        return np.argmax(q_values, axis=1)
