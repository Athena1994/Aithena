
from abc import abstractmethod

from aithena.qlearning.markov.state\
    import DictState, StateDescriptor, TensorDictState, arraydict_to_tensordict


from typing import Generic, TypeVar

T = TypeVar('T')
D = TypeVar('D')


class SimulationState(Generic[D, T]):

    def __init__(self, state_data: T, meta_data: D, terminal: bool):
        """Initializes the simulation state with scenario data."""
        self._state_data = state_data
        self._meta_data = meta_data
        self._terminal = terminal
        self._dict_state: DictState = None
        self._tensor_dict_state: TensorDictState = None

    # --- overwrite methods ---
    @abstractmethod
    def create_dict_state(self) -> StateDescriptor:
        """Returns the descriptor of the DictState."""
        raise NotImplementedError()

    # --- properties ---

    @property
    def is_terminal(self) -> bool:
        """Returns True if the simulation state is terminal, False otherwise."""
        return self._terminal

    @property
    def state_data(self) -> T:
        """Returns the scenario data of the simulation state."""
        return self._state_data

    @property
    def meta_data(self) -> D:
        """Returns the meta data of the simulation state."""
        return self._meta_data

    @property
    def dict_state(self) -> DictState:
        """Returns the current state of the simulation as a DictState."""
        if self._dict_state is None:
            self._dict_state = self.create_dict_state()
        return self._dict_state

    @property
    def tensor_dict_state(self) -> TensorDictState:
        """Returns the current state of the simulation as a DictState."""
        if self._tensor_dict_state is None:
            self._tensor_dict_state = arraydict_to_tensordict(self.dict_state)
        return self._tensor_dict_state
