
from abc import abstractmethod
from typing import Generic
from aithena.qlearning.simulation.simulation_state import T, D


class SimulationScenario(Generic[T]):

    def __init__(self):
        self._initial_state_data: T = None
        self._prepared: bool = False

    # --- overwrite these methods ---

    @abstractmethod
    def create_initial_state_data(self) -> T:
        raise NotImplementedError()

    async def on_prepare(self):
        pass

    def on_start_scenario(self, meta: D, init_state: T) -> T:
        return init_state

    # --- public methods ---

    async def prepare(self):
        await self.on_prepare()
        self._prepared = True

    # --- properties ---

    @property
    def prepared(self) -> bool:
        """Returns True if the scenario is prepared, False otherwise."""
        return self._prepared

    @property
    def initial_state_data(self) -> T:
        if self._initial_state_data is None:
            if not self._prepared:
                raise RuntimeError("Scenario must be prepared before accessing "
                                   "the initial state.")
            self._initial_state_data = self.create_initial_state_data()
        return self._initial_state_data
