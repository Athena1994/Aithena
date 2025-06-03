
import unittest

import numpy as np
import torch

from aithena.qlearning.helper import calculate_target_q
from aithena.qlearning.markov.experience import ExperienceBatch


class TestQLearningHelper(unittest.TestCase):
    class DummyTargetQNN(torch.nn.Module):
        def __init__(self, state_cnt, action_cnt):
            super().__init__()
            self.q_values = np.random.uniform(size=(state_cnt, action_cnt))

        def forward(self, x):
            res = torch.Tensor(self.q_values)[x.long()]
            return res

    def test_calculate_target_q(self):
        action_cnt = 4
        sample_cnt = 512
        state_cnt = 256

        nn = TestQLearningHelper.DummyTargetQNN(state_cnt, action_cnt)

        max_q_a = np.max(nn.q_values, axis=1)

        batch = ExperienceBatch(
            old_state={'a': torch.Tensor()},  # 512 x 4 -> 256 possible states
            action=None,
            reward=torch.Tensor(np.random.uniform(-100, 100, sample_cnt)),
            new_state=torch.Tensor(
                np.random.randint(0, state_cnt, (sample_cnt, ))),
            terminal=torch.Tensor(np.zeros(sample_cnt, dtype=bool)))

        discount_factor = 0.5

        expected_q = batch.rewards \
            + discount_factor * max_q_a[batch.new_states.view(-1).long()]

        batch.terminal[42] = True
        expected_q[42] = batch.rewards[42]
        batch.terminal[123] = True
        expected_q[123] = batch.rewards[123]
        batch.terminal[247] = True
        expected_q[247] = batch.rewards[247]

        target_q = calculate_target_q(nn, batch, discount_factor)
        self.assertTrue(np.allclose(list(expected_q), list(target_q)))
