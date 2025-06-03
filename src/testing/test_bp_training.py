

import unittest
import numpy as np
import torch
from aithena.torchwrapper.bptraining import BPTrainingConfig, BatchProvider


class MockProvider(BatchProvider):
    def __init__(self, data):
        self.data = data
        self.current_index = 0

    def provide_training_batch(self, batch_size: int):
        if self.current_index + batch_size >= len(self.data):
            self.current_index = 0

        batch_data = self.data[self.current_index:
                               self.current_index + batch_size].view(-1, 1)
        self.current_index += batch_size

        return batch_data, torch.sin(batch_data*3.14)


class TestBPTraining(unittest.TestCase):
    """Test by aproximating sinus function with a neural network."""

    def setUp(self):
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        cfg = {
            "optimizer": {
                "type": "adam",
                "learning-rate": 0.001,
                "weight-decay": 0.0001
            },
            "loss": {
                "type": "mse",
                "options": {}
            },
            "batch-cnt": 100,
            "batch-size": 32,
        }
        self.trainer = BPTrainingConfig(**cfg).create_trainer(self.nn)

        self.provider = MockProvider(torch.linspace(-1, 1, 1000))

    def test_training(self):
        for epoch in range(10):
            loss = self.trainer.train_epoch(self.provider)
            print(f"Epoch {epoch + 1}, Loss: {loss}")

        # Test the model
        self.trainer._model.eval()
        test_data = torch.tensor([[-1], [-0.5], [0], [0.5], [1]])
        predictions = self.trainer._model(test_data).detach().numpy()
        correct_predictions = torch.sin(test_data * 3.14).numpy()
        print("Predictions:", predictions)
        print("Correct:", correct_predictions)

        np.pow(correct_predictions - predictions, 2).mean(axis=0)
        self.assertLess(np.mean(np.pow(correct_predictions - predictions, 2)),
                        0.1,
                        "Model did not learn the sinus function well enough.")
