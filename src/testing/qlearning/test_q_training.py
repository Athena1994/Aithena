
import unittest

from torchrl.envs.libs.gym import GymEnv

from aithena.nn.dynamic_nn import DynamicNNConfig
from aithena.qlearning.arbiter.exploration_arbiter\
    import ExplorationArbiterConfig
from aithena.qlearning.dqn_trainer \
    import DQNTrainingConfig
from aithena.qlearning.simulation.markov_simulation import SimulationExperience
from jodisutils.misc.benchmark import Watch
from testing.qlearning.cartpole_sim \
    import CartPoleScenarioProvider, CartPoleSimulation, CartPoleState


class TestDQNTrainer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.cuda = True

        env = GymEnv("CartPole-v1")
        print(env.action_spec.shape)

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
            ).create_network(CartPoleState.state_descriptor, self.cuda)

        self.trainer = DQNTrainingConfig(**{
            "qlearning": {
                "replay-buffer-size": 10000,
                "discount-factor": 0.99,
                "state-descriptor": CartPoleState.state_descriptor,
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

        def reward(old_state, action, new_state: CartPoleState):
            return new_state.state_data.reward

        self.sim = CartPoleSimulation(reward, self.cuda)

        self.arbiter = ExplorationArbiterConfig(**{
            "type": "decaying-epsilon-greedy",
            "params": {
                "epsilon-start": 1,
                "epsilon-end": 0.05,
                "decay": 1000
            }
        }).create_arbiter(self.nn)

        self.provider = CartPoleScenarioProvider()

    def test_init(self):
        pass

    async def test_cart_pole(self):
        """Test the DQN trainer with the CartPole environment."""

        def optimize(exp: SimulationExperience):
            self.trainer.replay_buffer.add_experience(
                exp.dict_state_experience, 1)
            self.trainer.perform_training_step()

        async def episode(ix):
            with self.arbiter.exploration():
                self.provider.reset()
                experiences = await self.sim.run(self.provider,
                                                 self.arbiter, 1000,
                                                 optimize)
                return len(experiences)

        for ix in range(1000):
            w = Watch()
            w.start()
            cnt = await episode(ix)
            print(ix, w.elapsed(), w.elapsed()/cnt, cnt)
            if cnt == 500:
                break

        self.assertNotEqual(ix, 999)
