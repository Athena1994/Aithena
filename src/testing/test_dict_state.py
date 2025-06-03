import numpy as np
from aithena.qlearning.markov.state \
    import StateDescriptor, DictState, hash_state, make_descriptor, state_equal


import unittest


class TestDictState(unittest.TestCase):
    def setUp(self):
        self.state_descriptor = StateDescriptor({
            'a': (2,),
            'b': (4, 2),
            'c': (1, 3, 2)
        })
        self.state1 = DictState({
            'a': np.zeros((2,)),
            'b': np.zeros((4, 2)),
            'c': np.zeros((1, 3, 2))
        })
        self.state2 = DictState({
            'a': np.full((2,), 42),
            'b': np.full((4, 2), 42),
            'c': np.full((1, 3, 2), 42)
        })

        self.state3 = DictState({
            'a': np.full((2,), 42),
            'b': np.full((4, 2), 42),
            'c': np.full((1, 3, 2), 42)
        })

    def test_make_descriptor(self):
        desc = make_descriptor(self.state1)
        self.assertDictEqual(desc, {
            'a': (2,),
            'b': (4, 2),
            'c': (1, 3, 2)
        })

    def test_state_equal(self):
        self.assertTrue(state_equal(self.state1, self.state1))
        self.assertFalse(state_equal(self.state1, self.state2))
        self.assertFalse(state_equal(self.state2, self.state1))
        self.assertTrue(state_equal(self.state3, self.state2))

    def test_consistent_hash_values(self):
        hash1 = hash_state(self.state1)
        hash2 = hash_state(self.state2)
        hash3 = hash_state(self.state3)

        self.assertNotEqual(hash1, hash2)
        self.assertEqual(hash3, hash2)
