
from abc import abstractmethod
from typing import Generic


from aithena.qlearning.simulation.scenario import SimulationScenario
from aithena.qlearning.simulation.simulation_state import T, D


class ScenarioProvider(Generic[D, T]):

    def __init__(self,
                 scenarios: list[SimulationScenario[T]],
                 initial_meta_data: D):
        self._scenarios = scenarios

        self._iter: enumerate[SimulationScenario[T]] = None

        self._current_scenario: SimulationScenario[T] = None
        self._current_ix: int = -1

        self._autoreset = False  # reset _iter when the end is reached
        self._initial_meta_data = initial_meta_data

        self.reset()

    # --- overwrite methods ---
    @abstractmethod
    def on_start_scenario(self, prev_meta: D) -> D:
        """Hook for actions to perform when starting a new scenario."""
        raise NotImplementedError()

    # --- properties ---

    @property
    def autoreset(self) -> bool:
        return self._autoreset

    @autoreset.setter
    def autoreset(self, value: bool):
        self._autoreset = value

    @property
    def current_scenario(self) -> SimulationScenario[T]:
        """Returns the current scenario."""
        return self._current_scenario

    @property
    def current_ix(self) -> int:
        """Returns the index of the current scenario."""
        return self._current_ix

    # --- public methods ---

    def reset(self):
        """Resets the scenario generator to the beginning."""
        self._iter = enumerate(self._scenarios)
        self._current_ix = -1
        self._current_scenario = None

    async def start_next_scenario(self, current_meta: D | None) -> D:

        self._current_ix, self._current_scenario = next(self._iter, (-1, None))

        if self._current_scenario is None:
            if self._autoreset:
                self.reset()
                self._current_ix, self._current_scenario = next(self._iter)
            else:
                return None

        if current_meta is None:
            current_meta = self._initial_meta_data

        if not self._current_scenario.prepared:
            await self._current_scenario.prepare()

        return self.on_start_scenario(current_meta)

    # --- build-in methods ---
    def __len__(self) -> int:
        """Returns the number of scenarios in the provider."""
        return len(self._scenarios)
