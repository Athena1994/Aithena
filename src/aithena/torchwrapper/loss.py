

from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config

from torch import nn


@config
class LossConfig:
    type: str
    options: dict = Attribute(default={})

    def create_loss(self):
        """
        Create a loss function based on the type and options provided.
        :return: An instance of the created loss function.
        """
        loss_type = self.type.upper()
        if loss_type == 'MSE':
            return nn.MSELoss(**self.options)
        elif loss_type == 'L1':
            return nn.L1Loss(**self.options)
        elif loss_type == 'HUBER' or loss_type == 'SMOOTHL1':
            return nn.SmoothL1Loss(**self.options)
        elif loss_type == 'BCE':
            return nn.BCELoss(**self.options)
        elif loss_type == 'CE':
            return nn.CrossEntropyLoss(**self.options)
        else:
            raise ValueError(f"Unknown loss type: {self.type}")
