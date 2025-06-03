
import enum


from jodisutils.config.decorator import config
from torch.optim import Adam, Optimizer

from jodisutils.config.attribute import Attribute


class OptimizerTypes(enum.Enum):
    ADAM = 'adam'
    SGD = 'sgd'


@config
class OptimizerConfig:
    type: str
    learning_rate: float = Attribute('learning-rate')
    weight_decay: float = Attribute('weight-decay')

    def create_optimizer(self, parameters) -> Optimizer:
        if self.type == 'adam':
            return Adam(parameters,
                        lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.type}")
