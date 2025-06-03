import copy
from typing import Tuple

import torch
from torch import Tensor

from aithena.nn.dynamic_nn import DynamicNN
from aithena.qlearning.markov.experience import ExperienceBatch
from aithena.qlearning.helper import calculate_target_q
from aithena.qlearning.markov.replay_buffer import ReplayBuffer, StateDescriptor

from aithena.qlearning.target_update_strategy import \
      TargetUpdateStrategy, TargetUpdateStrategyConfig
from aithena.torchwrapper.bptraining \
    import BPTrainer, BPTrainingConfig, BatchProvider
from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config


@config
class DQNTrainingConfig:

    @config
    class QLearningConfig:
        discount_factor: float = Attribute('discount-factor')
        target_update_strategy: TargetUpdateStrategyConfig \
            = Attribute('target-update-strategy')
        replay_buffer_size: int = Attribute('replay-buffer-size')
        state_descriptor: StateDescriptor = Attribute('state-descriptor')

    training: BPTrainingConfig
    qlearning: QLearningConfig

    def create_trainer(self, dqn: torch.nn.Module, cuda: bool) -> 'DQNTrainer':

        bp_trainer = self.training.create_trainer(dqn)

        replay_buffer = ReplayBuffer(self.qlearning.replay_buffer_size,
                                     self.qlearning.state_descriptor,
                                     cuda)

        strat = self.qlearning.target_update_strategy.create_update_strategy()

        return DQNTrainer(
            cuda, dqn, replay_buffer, bp_trainer, strat,
            self.qlearning.discount_factor)


class DQNTrainer(BPTrainer, BatchProvider):
    """
        This class handles the training process for a Deep-Q-Network.

        The training is performed in episodes consisting of alternating
        exploration and training steps using a separated target dqn for
        determining the target Q-values. The target network parameters are
        updated after a minimum number of training samples.
    """

    def __init__(self, cuda: bool, dqn: DynamicNN, replay_buffer: ReplayBuffer,
                 bp_trainer: BPTrainer, update_strategy: TargetUpdateStrategy,
                 discount_factor: float,):
        self._policy_network: DynamicNN = dqn
        self._target_network: DynamicNN = copy.deepcopy(dqn)

        self._replay_buffer = replay_buffer
        self._bp_trainer = bp_trainer

        self._use_cuda = cuda
        self._discount_factor = discount_factor

        self._target_update_strategy: TargetUpdateStrategy = update_strategy

    # --- properties ---
    @property
    def dqn(self) -> DynamicNN:
        """Returns the current DQN model."""
        return self._policy_network

    @property
    def target_dqn(self) -> DynamicNN:
        """Returns the target DQN model."""
        return self._target_network

    @property
    def replay_buffer(self) -> ReplayBuffer:
        """Returns the replay buffer used for training."""
        return self._replay_buffer

    @property
    def bp_trainer(self) -> BPTrainer:
        """Returns the BPTrainer instance used for training."""
        return self._bp_trainer

    # --- public methods ---

    def perform_training_step(self) -> float:

        loss = self.bp_trainer.train_epoch(self, self._calc_q_values,
                                           self._after_training_batch)

        return loss

    # --- private methods ---

    def _calc_q_values(self, nn: DynamicNN, batch: ExperienceBatch) -> Tensor:

        q_values = nn(batch.old_states)
        selected_action_q = q_values[torch.arange(self.bp_trainer.batch_size),
                                     batch.actions]

        return selected_action_q

    def _after_training_batch(self, _: int, __: int, batch_size: int) -> None:
        self._update_target_network(batch_size)

    def _update_target_network(self, batch_size: int):
        self._target_update_strategy.update(self._policy_network,
                                            self._target_network)

    def provide_training_batch(self, batch_size: int) \
            -> Tuple[ExperienceBatch, torch.Tensor]:
        """
        Provides a training batch from the replay buffer.
        :return: A tuple containing an ExperienceBatch and a tensor of weights.
        """
        if self._replay_buffer.is_empty:
            raise RuntimeError("Replay buffer is empty.")

        if self._replay_buffer.size < batch_size:
            return None

        batch = self._replay_buffer.sample_experiences(batch_size, True)

        target_q = calculate_target_q(
            self._target_network, batch, self._discount_factor)

        return batch, target_q
