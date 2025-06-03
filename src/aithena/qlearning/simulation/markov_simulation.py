

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Type, TypeAlias, TypeVar

from aithena.qlearning.arbiter.qarbiter import QArbiter
from aithena.qlearning.markov.experience import Experience
from aithena.qlearning.markov.state import move_tensor_state
from aithena.qlearning.simulation.scenario_provider import ScenarioProvider
from aithena.qlearning.simulation.simulation_state \
    import SimulationState, T, D


@dataclass
class SimulationExperience:
    old_state: SimulationState[D, T]
    action: int
    reward: float
    new_state: SimulationState[D, T]
    terminal: bool

    @property
    def dict_state_experience(self) -> Experience:
        return Experience(
            self.old_state.dict_state,
            self.action,
            self.reward,
            self.new_state.dict_state,
            self.terminal
        )


RewardCallback: TypeAlias = Callable[[SimulationState[D, T],
                                      int,
                                      SimulationState[D, T]], float]
ExperienceWeightCallback: TypeAlias = Callable[[SimulationExperience], float]

TS = TypeVar('TS', bound=SimulationState)


class MarkovSimulation:

    def __init__(self, reward_callback: RewardCallback, state_type: Type[TS],
                 cuda: bool):
        self._state_type = state_type
        self._state: SimulationState = None
        self._reward_callback = reward_callback
        self._cuda = cuda

    # --- overwrite methods ---
    @abstractmethod
    async def advance_state(self,
                            meta: D,
                            state: T,
                            action: int) -> SimulationState:
        """Advances the current state of the simulation."""
        raise NotImplementedError()

    async def on_reset(self):
        pass

    # --- properties ---
    @property
    def current_scenario_finished(self) -> bool:
        """Returns True if the current scenario is finished, False otherwise."""
        return self._state.is_terminal if self._state else True

    # --- public methods ---
    async def reset(self) -> SimulationState:
        """Resets the simulation to its initial state."""
        self._state = None
        await self.on_reset()

    async def run(self, scenario_provider: ScenarioProvider, agent: QArbiter,
                  steps: int,
                  callback: Callable[[SimulationExperience], None] = None)\
            -> List[SimulationExperience]:
        """Runs the simulation for a given number of episodes."""

        experiences = []
        while len(experiences) < steps:
            if self.current_scenario_finished:
                if not await self._prepare_next_scenario(scenario_provider):
                    break

            exp = await self._perform_step(agent)
            if callback is not None:
                callback(exp)
            experiences.append(exp)

        return experiences

    # --- private methods ---

    async def _prepare_next_scenario(self, scenario_provider: ScenarioProvider)\
            -> bool:

        meta_data = self._state.meta_data if self._state else None
        meta_data = await scenario_provider.start_next_scenario(meta_data)

        if meta_data is None:
            return False

        scenario = scenario_provider.current_scenario

        initial_state_data = scenario.on_start_scenario(
            meta_data, scenario.initial_state_data)

        self._state = self._state_type(initial_state_data, meta_data, False)

        return True

    async def _perform_step(self, agent: QArbiter) \
            -> SimulationExperience:
        old_state = self._state
        if old_state is None:
            raise RuntimeError("Simulation state is not initialized.")

        if old_state.is_terminal:
            raise RuntimeError("Cannot perform step on terminal state.")

        action: int = agent.decide(
            move_tensor_state(old_state.tensor_dict_state, self._cuda))[0]

        self._state = await self.advance_state(
            old_state.meta_data, old_state.state_data, action)

        reward = self._reward_callback(old_state, action, self._state)

        return SimulationExperience(old_state, action, reward, self._state,
                                    self._state.is_terminal)
