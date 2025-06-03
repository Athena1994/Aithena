
from dataclasses import dataclass
import numpy as np
from aithena.qlearning.arbiter.qarbiter import QArbiter
from aithena.qlearning.markov.state import DictState, StateDescriptor
from aithena.qlearning.q_function import QFunction
from aithena.qlearning.simulation.markov_simulation import MarkovSimulation
from aithena.qlearning.simulation.scenario import SimulationScenario
from aithena.qlearning.simulation.scenario_provider import ScenarioProvider
from aithena.qlearning.simulation.simulation_state import SimulationState


@dataclass
class MetaData:
    cnt: int


@dataclass
class SimData:
    acc: float
    next_ix: int
    values: np.ndarray
    last_action: int = -1


class MockSimulationState(SimulationState[MetaData, SimData]):

    def create_dict_state(self) -> DictState:
        """Converts the mock state data to a DictState."""
        return {
            'a': np.array([self.state_data.acc,
                           self.state_data.next_ix,
                           self.state_data.last_action]),
            'b': np.array([self.meta_data.cnt]),
        }

    state_descriptor: StateDescriptor = {'a': (3, ), 'b': (1, )}


class MockScenario(SimulationScenario[SimData]):
    def __init__(self, values: np.ndarray):
        super().__init__()
        self._values = values

    def create_initial_state_data(self):
        return SimData(
            acc=self._values[0],
            next_ix=1,
            values=self._values
        )


class MockScenarioProvider(ScenarioProvider[MetaData, SimData]):
    def __init__(self, values: list[np.ndarray]):
        super().__init__(
            [MockScenario(v) for v in values],
            MetaData(cnt=0)
        )

    def on_start_scenario(self, prev_meta: MetaData) -> MetaData:
        return MetaData(cnt=prev_meta.cnt + 1)


class EnumAction:
    ADD = 0
    MULTIPLY = 1


class MockSimulation(MarkovSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(state_type=MockSimulationState, cuda=False,
                         *args, **kwargs)

    async def advance_state(self, meta: MetaData, state: SimData, action: int):
        val = state.values[state.next_ix]
        if action == EnumAction.ADD:
            acc = state.acc + val
        elif action == EnumAction.MULTIPLY:
            acc = state.acc * val
        else:
            raise ValueError(f"Unknown action: {action}")

        next_ix = state.next_ix + 1

        return MockSimulationState(
            SimData(acc, next_ix, state.values, last_action=action),
            MetaData(meta.cnt),
            terminal=(next_ix >= len(state.values))
        )


class MockAgent(QArbiter, QFunction):

    def __init__(self):
        super().__init__(q_fct=self)

    def get_q_values(self, state: DictState) -> np.ndarray:
        """Mock Q-values for the state."""
        # For simplicity, return a fixed array of Q-values
        if state['a'][2] != 0:
            return np.array([[1.0, 0.0]])
        return np.array([[0.0, 1.0]])

    def make_decision(self, q_values: np.ndarray) -> np.array:
        """Mock decision-making logic."""
        if q_values[0, 0] > q_values[0, 1]:
            return np.array([EnumAction.ADD])
        return np.array([EnumAction.MULTIPLY])
