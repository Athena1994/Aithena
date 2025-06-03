

from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config

from torch.nn import Module


@config
class TargetUpdateStrategyConfig:
    type: str
    params: dict

    def create_update_strategy(self) -> 'TargetUpdateStrategy':
        if self.type == 'delayed':
            config = DelayedUpdateStrategyConfig(**self.params)
            return DelayedUpdateStrategy(config.update_freq)

        elif self.type == 'soft':
            config = SoftUpdateStrategyConfig(**self.params)
            return SoftUpdateStrategy(config.tau)

        else:
            raise ValueError("Unknown target update strategy "
                             f"type: {self.type}")


class TargetUpdateStrategy:
    """
    Base class for target update strategies in Q-learning.
    """

    def update(self, policy_network: Module, target_network: Module):
        """
        Update the target network parameters based on the Q-network parameters.

        Args:
            q_network: The current Q-network.
            target_network: The target network to be updated.
        """
        raise NotImplementedError()


@config
class DelayedUpdateStrategyConfig:
    update_freq: int = Attribute('update-freq')


class DelayedUpdateStrategy(TargetUpdateStrategy):
    def __init__(self, update_freq: int):
        self._update_freq = update_freq
        self._update_counter = 0

    def update(self, policy_network: Module, target_network: Module):
        """
        Update the target network parameters based on the Q-network parameters
        if the update frequency condition is met.
        """
        self._update_counter += 1
        if self._update_counter >= self._update_freq:
            target_network.load_state_dict(policy_network.state_dict())
            self._update_counter = 0


@config
class SoftUpdateStrategyConfig:
    tau: float


class SoftUpdateStrategy(TargetUpdateStrategy):
    def __init__(self, tau: float):
        self._tau = tau

    def update(self, policy_network: Module, target_network: Module):
        """
        Update the target network parameters using a soft update approach.
        """
        for target_param, param in zip(target_network.parameters(),
                                       policy_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._tau) + param.data * self._tau)
