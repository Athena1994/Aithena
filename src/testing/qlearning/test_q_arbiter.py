
import unittest

import numpy as np
import torch

from aithena.qlearning.arbiter.decaying_epsilon_greedy_arbiter \
    import DecayingEpsilonGreedyArbiter
from aithena.qlearning.arbiter.epsilon_greedy_arbiter\
      import EpsilonGreedyArbiter
from aithena.qlearning.arbiter.exploration_arbiter\
      import ExplorationArbiterConfig
from aithena.qlearning.q_function import DeepQFunction


class TestDeepQFunction(unittest.TestCase):

    def test_normal_input(self):
        nn = torch.nn.Identity()
        qf = DeepQFunction(nn)

        test_data = np.random.uniform(-100, 100, size=(64, 8))
        for d in test_data:
            q_vals = qf.get_q_values(torch.Tensor(d))
            self.assertTrue(np.allclose(d, q_vals),
                            f"exp: {d}, given: {q_vals}")

        self.assertTrue(np.allclose(test_data,
                                    qf.get_q_values(torch.Tensor(test_data))))

        qf = DeepQFunction(nn)
        self.assertTrue(np.allclose(test_data,
                                    qf.get_q_values(torch.Tensor(test_data))))


class TestQArbiter(unittest.TestCase):
    def test_epsilon_greedy(self):
        np.random.seed(42)
        test_data = np.array([[0, 1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5, 3],
                             [2, 3, 4, 5, 0, 1],
                             [0, 1, 6, 3, 4, 5],
                             [0, 1, 2, 3, 8, 5]])
        expected_actions = [5, 4, 3, 2, 4]
        explore_prob = 0.2

        nn = torch.nn.Identity()
        qf = DeepQFunction(nn)

        qa = ExplorationArbiterConfig(**{
            'type': 'epsilon-greedy',
            'params': {'epsilon': explore_prob}
        }).create_arbiter(qf)

        self.assertTrue(isinstance(qa, EpsilonGreedyArbiter))
        self.assertEqual(qa.epsilon, explore_prob)

        qa = EpsilonGreedyArbiter(qf, epsilon=explore_prob)

        # test without explore
        with qa.exploration(False):
            actions = list(qa.decide(torch.Tensor(test_data)))
        self.assertListEqual(expected_actions, actions)

        # test with explore
        test_data = np.random.uniform(-100, 100, (10000, 10))
        expected_actions = np.argmax(test_data, axis=1)
        with qa.exploration():
            selected_actions = qa.decide(torch.Tensor(test_data))
        misses = np.sum(expected_actions != selected_actions)
        self.assertAlmostEqual(misses/test_data.shape[0],
                               explore_prob*0.9, 2)

    def test_decaying_epsilon_greedy(self):
        np.random.seed(42)
        test_data = np.array([[0, 1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5, 3],
                             [2, 3, 4, 5, 0, 1],
                             [0, 1, 6, 3, 4, 5],
                             [0, 1, 2, 3, 8, 5]])
        expected_actions = [5, 4, 3, 2, 4]

        decay = 10

        cfg = ExplorationArbiterConfig(**{
            'type': 'decaying-epsilon-greedy',
            'params': {
                'epsilon-start': 0.9,
                'epsilon-end': 0.1,
                'decay': decay
            }})

        nn = torch.nn.Identity()

        qa = cfg.create_arbiter(nn)
        self.assertTrue(isinstance(qa, DecayingEpsilonGreedyArbiter))
        self.assertEqual(qa.epsilon, 0.9)

        # test without explore
        with qa.exploration(False):
            actions = list(qa.decide(torch.Tensor(test_data)))
        self.assertListEqual(expected_actions, actions)

        action_count = 100

        # test with explore
        test_data = np.random.uniform(-100, 100, (50000, action_count))
        expected_actions = np.argmax(test_data, axis=1)

        for i in range(20):
            epsilon = qa.epsilon
            self.assertEqual(qa.exploration_index, i)
            self.assertAlmostEqual(epsilon,
                                   0.1 + (0.9 - 0.1) * np.exp(-i / decay), 2)

            with qa.exploration():
                selected_actions = list(qa.decide(torch.Tensor(test_data)))

            self.assertEqual(qa.exploration_index, i+1)
            self.assertEqual(qa.epsilon,
                             0.1 + (0.9 - 0.1) * np.exp(-(i+1) / decay))

            misses = np.sum(expected_actions != selected_actions)
            self.assertAlmostEqual(misses/test_data.shape[0],
                                   epsilon*(action_count-1)/action_count, 2)
