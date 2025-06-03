

from typing import Callable, Generic, Tuple, TypeAlias, TypeVar
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from aithena.torchwrapper.loss import LossConfig
from aithena.torchwrapper.optimizer import OptimizerConfig
from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config


@config
class BPTrainingConfig:
    optimizer: OptimizerConfig = Attribute('optimizer')
    loss: LossConfig = Attribute('loss')

    batch_cnt: int = Attribute('batch-cnt')
    batch_size: int = Attribute('batch-size')

    def create_trainer(self, model: torch.nn.Module) -> 'BPTrainer':
        """
        Creates a BPTrainer instance from the configuration.
        :param model: The model to be trained.
        :return: An instance of BPTrainer.
        """
        optimizer = self.optimizer.create_optimizer(model.parameters())
        loss_fn = self.loss.create_loss()
        return BPTrainer(model, optimizer, loss_fn, self.batch_cnt,
                         self.batch_size)


T = TypeVar('T')


class BatchProvider(Generic[T]):

    def provide_training_batch(self, batch_size: int) -> Tuple[T, torch.Tensor]:
        raise NotImplementedError()


FFCallback: TypeAlias = Callable[[torch.nn.Module, T], torch.Tensor]
StepCallback: TypeAlias = Callable[[int, int, int], None]


class BPTrainer(Generic[T]):

    def __init__(self, model: torch.nn.Module, optimizer: Optimizer,
                 loss: _Loss, batch_cnt: int, batch_size: int):
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss

        self._batch_cnt = batch_cnt
        self._batch_size = batch_size

        self._batch_ix = 0

    # --- properties ---
    @property
    def batch_cnt(self) -> int:
        """Returns the number of batches to train."""
        return self._batch_cnt

    @property
    def batch_size(self) -> int:
        """Returns the size of each training batch."""
        return self._batch_size

    @property
    def current_batch_ix(self) -> int:
        """Returns the index of the current batch."""
        return self._batch_ix

    # --- public methods ---

    def train_epoch(
        self,
        provider: BatchProvider,
        feed_forward_callback: FFCallback = lambda m, t: m(t),
        after_step_callback: StepCallback = lambda _, __, ___: None)\
            -> float:
        total_loss = 0.0

        self._model.train()
        self._model.requires_grad_(True)
        torch.set_grad_enabled(True)

        for self._batch_ix in range(self._batch_cnt):
            data = provider.provide_training_batch(self.batch_size)
            if data is None:
                break

            batch, target = data

            loss = self._training_step(batch, target, feed_forward_callback)

            after_step_callback(self._batch_ix,
                                self._batch_cnt,
                                self._batch_size)

            total_loss += loss

        torch.set_grad_enabled(False)
        self._model.requires_grad_(False)
        self._model.eval()

        return total_loss / self._batch_cnt

    # --- private methods ---

    def _training_step(self, batch: T, target: torch.Tensor,
                       ff_callback: FFCallback) -> float:

        self._optimizer.zero_grad()
        output = ff_callback(self._model, batch)
        loss = self._loss_fn(output, target)
        loss.backward()
        self._optimizer.step()

        return loss.item() / self._batch_size
