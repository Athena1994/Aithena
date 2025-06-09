
from aithena.qlearning.markov.state import DictState, TensorDictState


from typing import Callable, Generic, TypeAlias, TypeVar

from aithena.qlearning.simulation.scenario import ScenarioContext

T = TypeVar('T')
D = TypeVar('D')


DictStateFactory: TypeAlias = Callable[[D, T], DictState]


class SimulationState(Generic[D, T]):

    def __init__(self,
                 context: ScenarioContext,
                 observation: D,
                 episode_data: T,
                 terminal: bool,
                 tensor_state_factory: DictStateFactory):
        """Initializes the simulation state with scenario data."""
        self._episode_data = episode_data
        self._scenario_context = context
        self._observation = observation
        self._terminal = terminal
        self._tensor_state: TensorDictState = None
        self._dict_state_factory = tensor_state_factory

    # --- properties ---

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    @property
    def context(self) -> ScenarioContext:
        return self._scenario_context

    @property
    def episode_data(self) -> T:
        return self._episode_data

    @property
    def observation(self) -> D:
        return self._observation

    @property
    def tensor_state(self) -> TensorDictState:
        if self._tensor_state is None:
            if self._observation is None:
                return None
            self._tensor_state = self._dict_state_factory(
                self.observation, self.episode_data)
        return self._tensor_state
