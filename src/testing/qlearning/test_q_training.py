
from typing import List
import unittest

from aithena.nn.dynamic_nn import DynamicNNConfig
from aithena.qlearning.arbiter.exploration_arbiter\
    import ExplorationArbiterConfig
from aithena.qlearning.dqn_trainer \
    import DQNTrainingConfig
from aithena.qlearning.simulation.markov_simulation import SimulationExperience
from aithena.qlearning.simulation.simulation_state import SimulationState
from jodisutils.misc.benchmark import Watch
from testing.qlearning.cartpole_sim \
    import CartPoleScenario, CartPoleSimulation, EpisodeData, Observation


class TestDQNTrainer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.cuda = True

        def reward(old_state, action,
                   new_state: SimulationState[Observation, EpisodeData]):
            return new_state.episode_data.reward

        self.sim = CartPoleSimulation(reward, self.cuda)

        self.nn = DynamicNNConfig(**{
            'units': [{
                'name': 'in',
                "input-tags": ['state'],
                'nn': {
                    'type': 'sequential',
                    'options': {'layers': [
                        {'type': 'Linear', 'options': {'size': 128}},
                        {'type': 'ReLU'},
                        {'type': 'Linear', 'options': {'size': 128}},
                        {'type': 'ReLU'},
                        {'type': 'Linear', 'options': {'size': 2}}
                    ]}
                },
            }],
            'output-tag': 'in'}
            ).create_network(self.sim.tensor_state_descriptor, self.cuda)

        self.trainer = DQNTrainingConfig(**{
            "qlearning": {
                "replay-buffer-size": 10000,
                "discount-factor": 0.99,
                "state-descriptor": self.sim.tensor_state_descriptor,
                "target-update-strategy": {
                    "type": "soft",
                    "params": {"tau": 0.05}
                }
            },
            "training": {
                "optimizer": {
                    "type": "adam",
                    "learning-rate": 0.0001,
                    "weight-decay": 0
                },
                "loss": {"type": "MSE"},
                "batch-cnt": 1,
                "batch-size": 128,
            }
        }).create_trainer(self.nn, self.cuda)

        self.arbiter = ExplorationArbiterConfig(**{
            "type": "decaying-epsilon-greedy",
            "params": {
                "epsilon-start": 1,
                "epsilon-end": 0.05,
                "decay": 1000
            }
        }).create_arbiter(self.nn)

        self.scenario = CartPoleScenario()

    def test_init(self):
        pass

    async def test_cart_pole(self):
        """Test the DQN trainer with the CartPole environment."""

        def optimize(exp: List[SimulationExperience]):
            e = exp[0]
            self.trainer.replay_buffer.add_experience(
                e.as_experience, 1)
            self.trainer.perform_training_step()

        it = iter(self.scenario)

        async def episode(ix):
            with self.arbiter.exploration():
                await self.sim.reset(next(it))
                cnt, _, _ = await self.sim.run_episode(self.arbiter, -1)

            return cnt

        self.sim.exploration_callback.reset(optimize, 1)

        max_episodes = 110
        for ix in range(max_episodes):
            w = Watch()
            w.start()
            cnt = await episode(ix)
            print(ix, w.elapsed(), w.elapsed()/cnt, cnt)
            if cnt == 500:
                break

        self.assertNotEqual(ix, max_episodes - 1,)
