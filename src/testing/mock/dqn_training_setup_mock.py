
from asyncio import sleep
from torch import tensor
import torch

from aithena.qlearning.simulation.markov_simulation import MarkovSimulation
from aithena.qlearning.simulation.scenario import Scenario, ScenarioContext


class ListMockContext(ScenarioContext):

    def __init__(self, ar: list[int]):
        self.ar = ar
        self.ix = 0
        self.ep = -1

    def has_next(self) -> bool:
        return self.ep < len(self.ar)-1 or self.ix < self.ar[self.ep]-1

    def step(self):
        self.ix += 1
        if self.ix > self.ar[self.ep]:
            raise RuntimeError()
        return (self.ep, self.ix, self.ix == self.ar[self.ep]-1)

    def begin_episode(self):
        self.ep += 1
        self.ix = 0
        return (self.ep, self.ix, False), ""


class ListMockScenario(Scenario):
    def __init__(self, ar: list[int]):
        self.ar = ar

    def create_context(self) -> ScenarioContext:
        return ListMockContext(self.ar)


class MockSimulation(MarkovSimulation):
    def create_tensor_state(self, obs, data, device):
        if obs is None or data is None:
            raise ValueError("Observation and data must not be None")
        return tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=device)

    def get_tensor_state_descriptor(self):
        return {'a': (5,)}

    def __init__(self, cuda):
        super().__init__(lambda _, __, ___: 1, cuda)
        self.delay = 0

    async def advance_state(self, obs, state, action):
        # Mock implementation for testing purposes
        await sleep(self.delay)
        return "next_obs", obs[2]


class MockReplayBuffer:

    def add_experience(self, experience, weight: float) -> None:
        pass

    def add_experiences(self, _, __):
        pass


class MockTrainer:
    def __init__(self):
        self.called: int = 0
        self.replay_buffer = MockReplayBuffer()

    def perform_training_step(self) -> float:
        # Mock implementation for testing purposes
        self.called += 1
        return 0.0
