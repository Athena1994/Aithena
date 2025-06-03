
import unittest

import numpy as np

from testing.simulation.mock_simulation \
    import MetaData, MockSimulationState, SimData


class TestState(unittest.TestCase):
    def setUp(self):
        self.state = MockSimulationState(
            SimData(acc=0.1,
                    next_ix=5,
                    values=np.array([1, 2, 3])),
            MetaData(cnt=4),
            terminal=True)

    def test_initialization(self):
        self.assertEqual(self.state.state_data.acc, 0.1)
        self.assertEqual(self.state.state_data.next_ix, 5)
        self.assertTrue((self.state.state_data.values == [1, 2, 3]).all())

        self.assertEqual(self.state.meta_data.cnt, 4)

        self.assertTrue(self.state.is_terminal)

    def test_to_dict_state(self):
        call_cnt = 0

        def wrap_to_dict_state() -> dict:
            old_to_dict_state = self.state.create_dict_state

            def wrapper() -> dict:
                nonlocal call_cnt
                call_cnt += 1
                return old_to_dict_state()
            self.state.create_dict_state = wrapper

        wrap_to_dict_state()

        dict_state = self.state.dict_state
        self.assertEqual(dict_state['a'].shape, (3,))
        self.assertEqual(dict_state['b'].shape, (1,))
        self.assertTrue(np.all(dict_state['a'] == [0.1, 5, -1]))
        self.assertTrue(np.all(dict_state['b'] == [4]))

        dict_state = self.state.dict_state
        self.assertEqual(dict_state['a'].shape, (3,))
        self.assertEqual(dict_state['b'].shape, (1,))
        self.assertTrue(np.all(dict_state['a'] == [0.1, 5, -1]))
        self.assertTrue(np.all(dict_state['b'] == [4]))

        self.assertEqual(call_cnt, 1)
