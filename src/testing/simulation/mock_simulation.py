
from dataclasses import dataclass
import numpy as np
from torch import tensor
from aithena.qlearning.arbiter.qarbiter import QArbiter
from aithena.qlearning.markov.state \
    import DictState, TensorDictState
from aithena.qlearning.q_function import QFunction
from aithena.qlearning.simulation.markov_simulation import MarkovSimulation
from aithena.qlearning.simulation.scenario import Scenario, ScenarioContext


@dataclass
class SimObservation:
    value: float


@dataclass
class SimData:
    acc: float
    last_action: int = -1


class MockContext(ScenarioContext):
    def __init__(self, ar: np.ndarray):
        self._ar = ar
        self._ix = 0

    def step(self):
        """Advances the context to the next index."""
        if self.has_next():
            self._ix += 1
            return SimObservation(self._ar[self._ix])
        else:
            raise RuntimeError("No more steps available in the context.")

    def has_next(self) -> bool:
        """Checks if there are more steps available in the context."""
        return self._ix < len(self._ar) - 1

    def begin_episode(self):
        """Creates the initial state data for the context."""
        d = SimData(
            acc=self._ar[self._ix]
        )
        return self.step(), d


class MockScenario(Scenario):
    def __init__(self, values: np.ndarray):
        self._values = values

    def create_context(self) -> MockContext:
        return MockContext(self._values)


class EnumAction:
    ADD = 0
    MULTIPLY = 1


class MockSimulation(MarkovSimulation):

    def __init__(self, reward_callback: callable):
        super().__init__(cuda=False, reward_callback=reward_callback)

    def create_tensor_state(self, obs: SimObservation,
                            data: SimData, device) -> TensorDictState:
        """Converts the mock state data to a DictState."""
        return {
            'a': tensor(np.array([data.acc, float(data.last_action)]),
                        device=device),
            'b': tensor([obs.value], device=device),
        }

    def get_state_descriptor(self):
        return {'a': (2, ), 'b': (1, )}

    async def advance_state(self, obs: SimObservation,
                            state: SimData, action: int):

        if action == EnumAction.ADD:
            acc = state.acc + obs.value
        elif action == EnumAction.MULTIPLY:
            acc = state.acc * obs.value
        else:
            raise ValueError(f"Unknown action: {action}")

        return SimData(
            acc=acc,
            last_action=action
        ), acc > 100


class MockAgent(QArbiter, QFunction):

    def __init__(self):
        super().__init__(q_fct=self)

    def get_q_values(self, state: DictState) -> np.ndarray:
        """Mock Q-values for the state."""
        # For simplicity, return a fixed array of Q-values
        if state['a'][1] != 0:
            return np.array([[1.0, 0.0]])
        return np.array([[0.0, 1.0]])

    def make_decision(self, q_values: np.ndarray) -> np.array:
        """Mock decision-making logic."""
        if q_values[0, 0] > q_values[0, 1]:
            return np.array([EnumAction.ADD])
        return np.array([EnumAction.MULTIPLY])
