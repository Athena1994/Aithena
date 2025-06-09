

import enum

from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config


class TrainingSetupTypes(enum.Enum):
    DQN = 'dqn'


@config
class TrainingSetupConfig:
    options: dict
    type_: str = Attribute('type')

    def create(self, cuda: bool):
        type_ = self.type_.lower()
        if type_ == TrainingSetupTypes.DQN.value:
            from .dqn import DQNTrainingSetupConfig
            return DQNTrainingSetupConfig(**self.options).create(cuda)
        else:
            raise ValueError(f"Unknown training setup type: {type_}")


class BaseTrainingSetup:

    def __init__(self, cuda: bool):
        self._cuda = cuda

    @property
    def cuda(self) -> bool:
        return self._cuda

    async def run_epoch(self):
        raise NotImplementedError()
