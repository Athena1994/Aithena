
import unittest

import numpy as np

from aithena.qlearning.markov.replay_buffer import ReplayBuffer
from aithena.qlearning.simulation.simulation_state import SimulationState
from testing.simulation.mock_simulation \
    import (EnumAction, MetaData, MockAgent, MockScenarioProvider,
            MockSimulation, MockSimulationState, SimData)


def reward_transition(old_state: SimulationState[MetaData, SimData],
                      action: int,
                      new_state: SimulationState[MetaData, SimData]) -> float:
    """Calculates the reward based on the transition."""
    if action == 0:  # ADD
        return 1
    elif action == 1:  # MULTIPLY
        return -1
    else:
        raise ValueError("Invalid action")


class TestMarkovSimulation(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.sim = MockSimulation(reward_transition)

        self.buffer = ReplayBuffer(capacity=10,
                                   desc=MockSimulationState.state_descriptor)

        self.provider = MockScenarioProvider([
            np.array([1, 2, 3]),
            np.array([4, 5, 6, 7, 8])
        ])
        self.agent = MockAgent()

    async def test_run(self):
        """
            mock sim will alternate between ADD and MULTIPLY actions
            state: 'a': [acc, next_ix, last_action]
            meta: 'b': [cnt]
        """
        lst = await self.sim.run(self.provider, self.agent, 2)

        self.assertEqual(self.provider.current_ix, 0)
        self.assertTrue(self.sim.current_scenario_finished)

        self.assertEqual(len(lst), 2)
        lst = [e.dict_state_experience for e in lst]

        self.assertTrue(np.array_equal(lst[0].old_state['a'], [1, 1, -1]))
        self.assertTrue(np.array_equal(lst[0].new_state['a'], [3, 2, 0]))
        self.assertEqual(lst[0].old_state['b'], [1])
        self.assertEqual(lst[0].new_state['b'], [1])
        self.assertEqual(lst[0].action, EnumAction.ADD)
        self.assertEqual(lst[0].reward, 1)
        self.assertFalse(lst[0].terminal)

        self.assertEqual(lst[1].old_state['b'], [1])
        self.assertEqual(lst[1].new_state['b'], [1])
        self.assertTrue(np.array_equal(lst[1].old_state['a'], [3, 2, 0]))
        self.assertTrue(np.array_equal(lst[1].new_state['a'], [9, 3, 1]))
        self.assertEqual(lst[1].action, EnumAction.MULTIPLY)
        self.assertEqual(lst[1].reward, -1)
        self.assertTrue(lst[1].terminal)

        lst = await self.sim.run(self.provider, self.agent, 10)

        self.assertEqual(len(lst), 4)
        lst = [e.dict_state_experience for e in lst]

        self.assertEqual(lst[0].old_state['b'], [2])
        self.assertEqual(lst[0].new_state['b'], [2])
        self.assertTrue(np.array_equal(lst[0].old_state['a'], [4, 1, -1]))
        self.assertTrue(np.array_equal(lst[0].new_state['a'], [9, 2, 0]))

    async def test_run_auto_reset(self):
        self.provider.autoreset = True

        lst = await self.sim.run(self.provider, self.agent, 10)
        lst = [e.dict_state_experience for e in lst]

        self.assertEqual(len(lst), 10)

        self.assertEqual(lst[0].old_state['b'], [1])
        self.assertEqual(lst[0].new_state['b'], [1])
        self.assertTrue(np.array_equal(lst[0].old_state['a'], [1, 1, -1]))
        self.assertTrue(np.array_equal(lst[0].new_state['a'], [3, 2, 0]))

        self.assertEqual(lst[2].old_state['b'], [2])
        self.assertEqual(lst[2].new_state['b'], [2])
        self.assertTrue(np.array_equal(lst[2].old_state['a'], [4, 1, -1]))
        self.assertTrue(np.array_equal(lst[2].new_state['a'], [9, 2, 0]))

        self.assertEqual(lst[6].old_state['b'], [3])
        self.assertEqual(lst[6].new_state['b'], [3])
        self.assertTrue(np.array_equal(lst[6].old_state['a'], [1, 1, -1]))
        self.assertTrue(np.array_equal(lst[6].new_state['a'], [3, 2, 0]))

        self.assertEqual(lst[8].old_state['b'], [4])
        self.assertEqual(lst[8].new_state['b'], [4])
        self.assertTrue(np.array_equal(lst[8].old_state['a'], [4, 1, -1]))
        self.assertTrue(np.array_equal(lst[8].new_state['a'], [9, 2, 0]))
