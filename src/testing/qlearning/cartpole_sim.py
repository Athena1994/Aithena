
from dataclasses import dataclass

import torch
from aithena.qlearning.simulation.markov_simulation \
    import MarkovSimulation, RewardCallback
from aithena.qlearning.simulation.scenario import SimulationScenario
from aithena.qlearning.simulation.scenario_provider import ScenarioProvider
from aithena.qlearning.simulation.simulation_state import SimulationState


import gymnasium as gym


@dataclass
class CartPoleStateData:
    state: torch.Tensor
    reward: float


@dataclass
class CartPoleMeta:
    episode: int
    step: int
    env: gym.Env


class CartPoleState(SimulationState[CartPoleMeta, CartPoleStateData]):
    def create_dict_state(self) -> dict:
        """Converts the cartpole state data to a DictState."""
        return {'state': self.state_data.state}

    state_descriptor = {
        'state': (4,)
    }


class CartPoleScenarioProvider(ScenarioProvider[CartPoleMeta,
                                                CartPoleStateData]):
    def __init__(self):
        super().__init__([CartPoleScenario()],
                         CartPoleMeta(0, 0, None))

    def on_start_scenario(self, prev_meta: CartPoleMeta) -> CartPoleMeta:
        """Increments the episode count when starting a new scenario."""
        return CartPoleMeta(prev_meta.episode + 1, 0, None)


class CartPoleScenario(SimulationScenario[CartPoleMeta]):
    def __init__(self):
        super().__init__()
        self._env = gym.make("CartPole-v1")
#        GymEnv("CartPole-v1")

    def on_start_scenario(self, meta: CartPoleMeta,
                          init_state: CartPoleStateData) -> CartPoleStateData:
        state, info = self._env.reset()

        meta.env = self._env

        return CartPoleStateData(
            state=torch.tensor(state, dtype=torch.float32),
            reward=0.0
        )

    def create_initial_state_data(self) -> CartPoleStateData:
        """Creates the initial state data for the CartPole environment."""
        return CartPoleStateData(None, 0)


class CartPoleSimulation(MarkovSimulation):
    def __init__(self, reward: RewardCallback, cuda: bool):
        super().__init__(reward, CartPoleState, cuda)

    async def advance_state(self, meta: CartPoleMeta, state: CartPoleStateData,
                            action: int):
        obs, reward, terminated, truncated, _ = meta.env.step(action)

        return CartPoleState(
            state_data=CartPoleStateData(
                state=torch.tensor(obs, dtype=torch.float32),
                reward=reward
            ),
            meta_data=CartPoleMeta(
                episode=meta.episode,
                step=meta.step + 1,
                env=meta.env
            ),
            terminal=terminated or truncated
        )
