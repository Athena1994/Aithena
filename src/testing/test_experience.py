

import unittest

import numpy as np

from aithena.qlearning.markov.experience import Experience
from aithena.qlearning.markov.state import DictState


class TestExperience(unittest.TestCase):
    def setUp(self):

        state1 = DictState({
            'a': np.zeros((2,)),
            'b': np.zeros((4, 2)),
            'c': np.zeros((1, 3, 2))
        })
        state2 = DictState({
            'a': np.full((2,), 42),
            'b': np.full((4, 2), 42),
            'c': np.full((1, 3, 2), 42)
        })

        state3 = DictState({
            'a': np.full((2,), 42),
            'b': np.full((4, 2), 42),
            'c': np.full((1, 3, 2), 42)
        })

        self.experience1 = Experience(
            old_state=state1,
            action=3,
            reward=234.0,
            new_state=state2
        )

        self.experience2 = Experience(
            old_state=state2,
            action=5,
            reward=-12.0,
            new_state=state3
        )
        self.experience3 = Experience(
            old_state=state1,
            action=3,
            reward=234.0,
            new_state=state2
        )

    def test_hash(self):
        self.assertEqual(hash(self.experience1), hash(self.experience3))
        self.assertNotEqual(hash(self.experience1), hash(self.experience2))

    def test_equality(self):
        self.assertEqual(self.experience1, self.experience3)
        self.assertNotEqual(self.experience1, self.experience2)
        self.assertNotEqual(self.experience2, self.experience3)
