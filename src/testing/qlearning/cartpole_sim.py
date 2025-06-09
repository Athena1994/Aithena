
from dataclasses import dataclass

import torch
from aithena.qlearning.simulation.markov_simulation \
    import MarkovSimulation, RewardCallback


import gymnasium as gym

from aithena.qlearning.simulation.scenario import Scenario, ScenarioContext


@dataclass
class Observation:
    env: gym.Env


@dataclass
class EpisodeData:
    step: int
    episode: int
    state: torch.Tensor
    reward: float


class Context(ScenarioContext):
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self._episode = 0

    def begin_episode(self):
        """Initializes the environment and returns the first observation."""
        obs, info = self.env.reset()
        self._episode += 1
        return (Observation(self.env),
                EpisodeData(0, self._episode-1,
                            torch.tensor(obs, dtype=torch.float32), 0.0))

    def step(self):
        return Observation(self.env)

    def has_next(self):
        return True


class CartPoleScenario(Scenario):
    def create_context(self) -> Context:
        """Creates a new context for the CartPole scenario."""
        return Context()


class CartPoleSimulation(MarkovSimulation):
    def __init__(self, reward: RewardCallback, cuda: bool):
        super().__init__(reward, cuda)

    def create_tensor_state(self, obs: Observation, ep: EpisodeData,
                            device) -> dict:
        """Converts the cartpole state data to a DictState."""
        return {'state': ep.state.cuda() if device == 'cuda'
                else ep.state.cpu()}

    def get_tensor_state_descriptor(self):
        return {'state': (4,)}

    async def advance_state(self, obs: Observation, ep: EpisodeData,
                            action: int):

        obs, reward, terminated, truncated, info = obs.env.step(action)
        return (EpisodeData(
            ep.step + 1,
            ep.episode,
            torch.tensor(obs, dtype=torch.float32),
            reward
        ), terminated or truncated)
