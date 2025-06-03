

import unittest
import numpy as np
from testing.simulation.mock_simulation import MockScenario


class TestScenario(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scenario = MockScenario(np.array([1, 2, 3]))

    async def test_initial_state_data(self):
        initial_state = self.scenario.create_initial_state_data()
        self.assertEqual(initial_state.acc, 1)
        self.assertEqual(initial_state.next_ix, 1)
        self.assertTrue((initial_state.values == [1, 2, 3]).all())

        with self.assertRaises(RuntimeError):
            a = self.scenario.initial_state_data

        await self.scenario.prepare()

        a = self.scenario.initial_state_data
        self.assertEqual(a.acc, 1)
        self.assertEqual(a.next_ix, 1)
        self.assertTrue((a.values == [1, 2, 3]).all())

        b = self.scenario.initial_state_data
        self.assertEqual(a, b)

        self.assertIsNot(initial_state, a)
