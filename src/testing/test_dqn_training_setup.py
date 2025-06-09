

from asyncio import Event
import asyncio
from typing import List
import unittest

import torch

from aithena.qlearning.arbiter.exploration_arbiter \
    import ExplorationArbiterConfig
from aithena.qlearning.simulation.markov_simulation import SimulationExperience
from aithena.trainingsetups.dqn import DQNTrainingSetup, DQNTrainingSetupConfig
from testing.mock.dqn_training_setup_mock \
    import ListMockScenario, MockSimulation, MockTrainer


class TestDQNTrainingSetup(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.simulation = MockSimulation(False)

        self.tr_interval = 2

        self.trainer = MockTrainer()

        self.setup = DQNTrainingSetup(
            None, self.trainer, ExplorationArbiterConfig(
                'epsilon-greedy', {'epsilon': 0.1}).create_arbiter(
                    torch.nn.Linear(5, 2)), False,
            self.simulation, self.tr_interval
        )

    def test_initialization(self):
        pass

    def test_config(self):

        _ = DQNTrainingSetupConfig(**{
            "trainer": {
                'qlearning': {
                    'discount-factor': 0.99,
                    'target-update-strategy': {
                        'type': 'soft',
                        'params': {'tau': 0.005}
                    },
                    'replay-buffer-size': 32,
                    'state-descriptor': {'a': (42,)}
                },
                'training': {
                    'optimizer': {
                        'type': 'adam',
                        'learning-rate': 0.001,
                        'weight-decay': 0
                    },
                    'loss': {'type': 'MSE'},
                    'batch-cnt': 1,
                    'batch-size': 8
                }
            },
            'nn': {
                'units': [
                    {
                        'name': 'out',
                        'nn': {'type': 'Linear', 'options': {'size': 2}},
                        'input-tags': ['a']
                    }
                ],
                'output-tag': 'out'
            },
            'arbiter': {
                'type': 'decaying-epsilon-greedy',
                'params': {
                    'epsilon-start': 1,
                    'epsilon-end': 0.01,
                    'decay': 1000
                }
            },
            'optimize-after': 1
        }).create(False, self.simulation)

    async def test_simple_run(self):
        scenario = ListMockScenario([3, 4])

        mock_callback_order = [
            ((0, 0, False), (0, 1, False), False),
            ((0, 1, False), (0, 2, True), False),
            ((0, 2, True), None, True),

            ((1, 0, False), (1, 1, False), False),
            ((1, 1, False), (1, 2, False), False),
            ((1, 2, False), (1, 3, True), False),
            ((1, 3, True), None, True),
        ]

        cb_ix = 0

        def mock_callback(ex: List[SimulationExperience]):
            nonlocal cb_ix
            self.assertEqual(len(ex), 1)
            ex: SimulationExperience = ex[0]
            self.assertEqual(ex.old_state.observation,
                             mock_callback_order[cb_ix][0])
            self.assertEqual(ex.new_state.observation,
                             mock_callback_order[cb_ix][1])
            self.assertEqual(ex.terminal, mock_callback_order[cb_ix][2])
            cb_ix += 1

        self.setup.exploration_callback.reset(mock_callback, 1)

        episodes = await self.setup.run_epoch(scenario, True)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0], (3, 3))
        self.assertEqual(episodes[1], (4, 4))

    async def test_callbacks(self):

        scenario = ListMockScenario([3, 5])

        expected = [(3, True, 1), (3, False, 2), (2, True, 3)]

        cb_ix = 0

        def mock_callback(ex: List[SimulationExperience]):
            nonlocal cb_ix
            self.assertEqual(len(ex), expected[cb_ix][0])
            self.assertEqual(ex[-1].terminal, expected[cb_ix][1])
            self.assertEqual(self.trainer.called, expected[cb_ix][2])
            cb_ix += 1

        self.setup.exploration_callback.reset(mock_callback, 3)

        episodes = await self.setup.run_epoch(scenario, True)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0], (3, 3))
        self.assertEqual(episodes[1], (5, 5))

        self.assertEqual(self.trainer.called, 4)

    async def test_abort(self):

        scenario = ListMockScenario([3, 5])
        self.simulation.delay = 0.1

        abort = Event()

        async def abort_after_delay():
            await asyncio.sleep(0.4)
            abort.set()

        asyncio.create_task(abort_after_delay())

        episodes = await self.setup.run_epoch(scenario, True, abort)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0], (3, 3))

    async def test_max_episodes(self):

        scenario = ListMockScenario([3, 5, 3, 3, 3])

        episodes = await self.setup.run_epoch(scenario, True)
        self.assertEqual(len(episodes), 5)

        episodes = await self.setup.run_epoch(scenario, True, max_episodes=3)
        self.assertEqual(len(episodes), 3)

    async def test_validation(self):
        scenario = ListMockScenario([3, 5])

        episodes = await self.setup.run_epoch(scenario, False)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0], (3, 3))
        self.assertEqual(episodes[1], (5, 5))

        self.assertEqual(self.trainer.called, 0)
