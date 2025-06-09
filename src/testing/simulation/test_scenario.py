

import unittest
import numpy as np
from testing.simulation.mock_simulation import MockScenario


class TestScenario(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scenario = MockScenario(np.array([1, 2, 3, 4, 5]))
        self.context = self.scenario.create_context()

    async def test_initial_state_data(self):
        self.assertTrue(np.array_equal(self.context._ar,
                                       np.array([1, 2, 3, 4, 5])))
        self.assertEqual(self.context._ix, 0)
