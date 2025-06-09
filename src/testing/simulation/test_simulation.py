
import unittest

import numpy as np

from aithena.qlearning.simulation.simulation_state import SimulationState
from testing.simulation.mock_simulation \
    import (MockAgent, MockScenario, MockSimulation, SimData, SimObservation)


def reward_transition(old_state: SimulationState[SimObservation, SimData],
                      action: int,
                      new_state: SimulationState[SimObservation, SimData]
                      ) -> float:
    """Calculates the reward based on the transition."""
    if action == 0:  # ADD
        return 1
    elif action == 1:  # MULTIPLY
        return 2
    else:
        raise ValueError("Invalid action")


class TestMarkovSimulation(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.scenario = MockScenario(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.sim = MockSimulation(reward_transition)
        self.agent = MockAgent()

    async def test_run(self):
        """
            mock sim will alternate between ADD and MULTIPLY actions
            state: 'a': [acc, last_action]
            meta: 'b': [value]
        """

        with self.assertRaises(RuntimeError):
            _, _ = await self.sim.run_episode(self.agent, 2)

        for initial_context in iter(self.scenario):
            await self.sim.reset(initial_context)
            cnt, reward, full = await self.sim.run_episode(self.agent)

        it = iter(self.scenario)
        await self.sim.reset(next(it))
        cnt, reward, full = await self.sim.run_episode(self.agent, 4)
        self.assertEqual(reward, 6)
        self.assertEqual(cnt, 4)
        self.assertFalse(full)
        cnt, reward, full = await self.sim.run_episode(self.agent, 4)
        self.assertEqual(reward, 3)
        self.assertEqual(cnt, 2)
        self.assertTrue(full)

        await self.sim.reset(next(it))
        cnt, reward, full = await self.sim.run_episode(self.agent, 4)
        self.assertEqual(reward, 3)
        self.assertEqual(cnt, 2)
        self.assertTrue(full)

        await self.sim.reset(next(it))
        cnt, reward, full = await self.sim.run_episode(self.agent, 4)
        self.assertEqual(reward, 1)
        self.assertEqual(cnt, 1)
        self.assertTrue(full)

        self.assertIsNone(next(it, None))
