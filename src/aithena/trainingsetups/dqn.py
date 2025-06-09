
from asyncio import Event
from typing import Callable, List, Tuple, TypeAlias

from aithena.nn.dynamic_nn import DynamicNN, DynamicNNConfig
from aithena.qlearning.arbiter.exploration_arbiter \
    import ExplorationArbiter, ExplorationArbiterConfig
from aithena.qlearning.dqn_trainer import DQNTrainer, DQNTrainingConfig
from aithena.qlearning.simulation.markov_simulation \
    import MarkovSimulation, SimulationExperience
from aithena.qlearning.simulation.scenario import Scenario
from aithena.trainingsetups.basesetup import BaseTrainingSetup

from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config
from jodisutils.misc.callbacks import BufferedCallback, IntervalCallback


@config
class DQNTrainingSetupConfig:
    trainer: DQNTrainingConfig
    nn: DynamicNNConfig
    arbiter: ExplorationArbiterConfig

    optimize_after: int = Attribute('optimize-after', default=1)

    def create(self, cuda: bool, simulation: MarkovSimulation):
        nn = self.nn.create_network(self.trainer.qlearning.state_descriptor,
                                    cuda)
        trainer = self.trainer.create_trainer(nn, cuda)
        arbiter = self.arbiter.create_arbiter(nn)

        return DQNTrainingSetup(nn, trainer, arbiter, cuda, simulation,
                                self.optimize_after)


WeightCallback: TypeAlias = Callable[[SimulationExperience], float]


class DQNTrainingSetup(BaseTrainingSetup):

    def __init__(self, nn: DynamicNN, trainer: DQNTrainer,
                 arbiter: ExplorationArbiter, cuda: bool,
                 simulation: MarkovSimulation, training_interval: int):
        super().__init__(cuda)

        self._nn = nn
        self._trainer = trainer
        self._arbiter = arbiter
        self._simulation = simulation

        self._train = False

        self._expl_cb = BufferedCallback()
        self._weight_cb: WeightCallback = lambda _: 1.0
        self._optimization_cb = IntervalCallback(
            self._trainer.perform_training_step, interval=training_interval)

        self._exploration_callback = BufferedCallback()

        simulation.exploration_callback.reset(
            interval=1,
            callback=self._on_exploration_step
        )

    # --- properties ---

    @property
    def exploration_callback(self) -> BufferedCallback:
        """Returns the exploration callback."""
        return self._exploration_callback

    # --- public methods ---

    def set_weight_callback(self, callback: WeightCallback) -> None:
        if callback is None:
            raise ValueError("Weight callback cannot be None")
        self._weight_cb = callback

    async def run_epoch(self, scenario: Scenario, train: bool,
                        abort: Event = None, max_episodes: int = -1
                        ) -> List[Tuple[int, int]]:
        self._train = train

        result = []

        with self._arbiter.exploration():
            episode = 0
            for init_context in iter(scenario):
                episode += 1

                await self._simulation.reset(init_context)

                cnt, reward, terminal = await self._simulation.run_episode(
                    self._arbiter, -1, abort)

                if not terminal:
                    break

                result.append((cnt, reward))

                if ((max_episodes > 0 and episode >= max_episodes)
                        or (abort and not abort.is_set())):
                    break

            return result

    # --- private methods ---

    def _on_exploration_step(self, exp: List[SimulationExperience]) -> None:
        for e in exp:
            self._trainer.replay_buffer.add_experience(
                 e.as_experience, self._weight_cb(e))
            self.exploration_callback(e)

        if self._train:
            self._optimization_cb()
