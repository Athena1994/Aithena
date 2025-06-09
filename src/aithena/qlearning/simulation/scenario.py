
from typing import Generic, Tuple, TypeVar


D = TypeVar('D')
T = TypeVar('T')


class ScenarioContext(Generic[D, T]):

    def begin_episode(self) -> Tuple[D, T]:
        raise NotImplementedError()

    def step(self) -> D:
        raise NotImplementedError()

    def has_next(self) -> bool:
        return NotImplementedError()


class Scenario(Generic[D, T]):

    class Iterator:
        def __init__(self, context: ScenarioContext):
            self._context = context

        def __next__(self) -> Tuple[D, T, ScenarioContext]:
            if not self._context.has_next():
                raise StopIteration()
            return self._context

        def __iter__(self):
            return self

    def __iter__(self) -> Iterator:
        return Scenario.Iterator(self.create_context())

    # --- overwrite these methods ---
    def create_context(self) -> ScenarioContext:
        raise NotImplementedError()
