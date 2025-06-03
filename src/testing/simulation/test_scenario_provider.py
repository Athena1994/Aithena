

import unittest

import numpy as np
from testing.simulation.mock_simulation \
    import MockScenario, MockScenarioProvider


class TestScenarioProvider(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = MockScenarioProvider([
            np.array([1, 2, 3]),
            np.array([4, 5, 6, 7, 8])
        ])

    def test_initialization(self):
        self.assertEqual(len(self.provider), 2)
        self.assertFalse(self.provider.autoreset)
        self.assertEqual(self.provider.current_ix, -1)
        self.assertEqual(self.provider.current_scenario, None)
        self.assertFalse(self.provider._scenarios[0].prepared)

    async def test_start_next(self):
        self.assertEqual(self.provider.current_ix, -1)
        self.assertEqual(self.provider.current_scenario, None)

        meta = await self.provider.start_next_scenario(None)
        self.assertEqual(meta.cnt, 1)
        self.assertEqual(self.provider.current_ix, 0)
        self.assertIsInstance(self.provider.current_scenario, MockScenario)
        self.assertEqual(
            len(self.provider.current_scenario.initial_state_data.values), 3)

        meta = await self.provider.start_next_scenario(meta)
        self.assertEqual(meta.cnt, 2)
        self.assertEqual(self.provider.current_ix, 1)
        self.assertIsInstance(self.provider.current_scenario, MockScenario)
        self.assertEqual(
            len(self.provider.current_scenario.initial_state_data.values), 5)

        meta = await self.provider.start_next_scenario(meta)
        self.assertIsNone(meta)
        self.assertEqual(self.provider.current_ix, -1)
        self.assertEqual(self.provider.current_scenario, None)

        self.provider.autoreset = True

        meta = await self.provider.start_next_scenario(None)
        self.assertEqual(meta.cnt, 1)
        self.assertEqual(self.provider.current_ix, 0)
        self.assertIsInstance(self.provider.current_scenario, MockScenario)
        self.assertEqual(
            len(self.provider.current_scenario.initial_state_data.values), 3)
