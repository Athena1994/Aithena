

from abc import abstractmethod
from asyncio import Event
from dataclasses import dataclass
from typing import Callable, Generic, Tuple, TypeAlias

from aithena.qlearning.arbiter.qarbiter import QArbiter
from aithena.qlearning.markov.experience import Experience
from aithena.qlearning.markov.state import StateDescriptor, TensorDictState
from aithena.qlearning.simulation.scenario import ScenarioContext
from aithena.qlearning.simulation.simulation_state \
    import SimulationState, T, D
from jodisutils.misc.callbacks import BufferedCallback


@dataclass
class SimulationExperience(Generic[D, T]):
    old_state: SimulationState[D, T]
    action: int
    reward: float
    new_state: SimulationState[D, T]
    terminal: bool

    @property
    def as_experience(self) -> Experience:
        return Experience(
            self.old_state.tensor_state,
            self.action,
            self.reward,
            self.new_state.tensor_state,
            self.terminal
        )


RewardCallback: TypeAlias = Callable[[SimulationState[D, T],
                                      int,
                                      SimulationState[D, T]], float]


class MarkovSimulation:

    def __init__(self, reward_callback: RewardCallback, cuda: bool):
        if reward_callback is None:
            raise ValueError("Reward callback must not be None.")

        self._state: SimulationState = None
        self._reward_callback = reward_callback
        self._exploration_callback = BufferedCallback()
        self._cuda = cuda

    # --- overwrite methods ---

    @abstractmethod
    async def advance_state(self, observation: D, episode_data: T,
                            action: int) -> Tuple[T, bool]:
        """Advances the current state of the simulation."""
        raise NotImplementedError()

    @abstractmethod
    def create_tensor_state(self, obs: D, episode_data: T, cuda: bool
                            ) -> TensorDictState:
        """Converts the current state of the simulation to a DictState."""
        raise NotImplementedError()

    @abstractmethod
    def get_tensor_state_descriptor(self) -> StateDescriptor:
        """Returns the state descriptor for the tensor state."""
        raise NotImplementedError()

    async def on_reset(self):
        pass

    # --- properties ---

    @property
    def tensor_state_descriptor(self) -> StateDescriptor:
        return self.get_tensor_state_descriptor()

    @property
    def terminal_state(self) -> bool:
        """Returns True if the current scenario is finished, False otherwise."""
        return self._state.is_terminal if self._state else True

    @property
    def exploration_callback(self) -> BufferedCallback:
        """Returns the exploration callback."""
        return self._exploration_callback

    # --- public methods ---

    async def reset(self, context: ScenarioContext) -> None:
        """Resets the simulation to its initial state."""

        self.set_state(context, *context.begin_episode(), False)

        await self.on_reset()

    def set_state(self, context: ScenarioContext, obs: D, data: T,
                  terminal: bool):

        device = 'cuda' if self._cuda else 'cpu'

        def tensor_factory(obs: D, ep_data: T) -> TensorDictState:
            return self.create_tensor_state(obs, ep_data, device)

        self._state = SimulationState(
            context=context,
            observation=obs,
            episode_data=data,
            terminal=terminal,
            tensor_state_factory=tensor_factory)

    async def run_episode(self, agent: QArbiter, max_steps: int = -1,
                          abort: Event = None) -> Tuple[int, int, bool]:
        """Runs the simulation for a given number of episodes."""

        if self.terminal_state:
            raise RuntimeError(
                "Cannot run episode on terminal state. "
                "Please reset the simulation first.")

        step = 0
        total_reward = 0.0

        while ((not self.terminal_state and (max_steps < 0 or step < max_steps))
               and not (abort and abort.is_set())):

            context: ScenarioContext = self._state.context

            # determine action based on current state
            action: int = agent.decide(self._state.tensor_state)[0]

            # advance the state based on the action
            new_episode_data, terminal = await self.advance_state(
                self._state.observation, self._state.episode_data, action)

            old_state: SimulationState = self._state

            if not terminal and context.has_next():
                new_observation = context.step()
            else:
                new_observation = None
                terminal = True

            self.set_state(context, new_observation, new_episode_data, terminal)
            reward = self._reward_callback(old_state, action, self._state)

            self.exploration_callback(
                SimulationExperience(old_state, action, reward, self._state,
                                     self._state.is_terminal))

            step += 1
            total_reward += reward

        self.exploration_callback.flush()

        return step, total_reward, self.terminal_state
