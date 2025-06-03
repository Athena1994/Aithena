
from typing import Tuple
import torch

from aithena.nn.dynamic_nn import DynamicNN
from aithena.qlearning.markov.experience import ExperienceBatch


def perform_training_step(nn: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          loss: torch.nn.Module,
                          tr_data: Tuple[object, object]):
    input_data, target_data = tr_data

    optimizer.zero_grad()
    result_tensor = nn(input_data)
    loss_tensor = loss(result_tensor, target_data)
    loss_tensor.backward()
    optimizer.step()


def calculate_target_q(target_dqn: DynamicNN,
                       experience_batch: ExperienceBatch,
                       discount_factor: float) -> torch.Tensor:
    """
    Calculates the Q_action target for given experiences.
    :param target_dqn: Target DQN for Q-Value calculation.
    :param rewards: [N] Rewards for choosing 'next_states'
    :param next_states: [N] follow states (DQNWrapper input)
    :param discount_factor: reduces impact from future states
    :return: Tensor containing one q-value per experience tuple
    """
    next_q_values = target_dqn(experience_batch.new_states)
    max_q_next, _ = torch.max(next_q_values, dim=1)
    # Q_t = reward + gamma * max_a Q(s_{t+1}, a) (nont terminal state)
    # Q_t = reward (terminal state)

    terminal_mask = experience_batch.terminal.float()
    return experience_batch.rewards \
        + discount_factor * max_q_next * (1 - terminal_mask)
