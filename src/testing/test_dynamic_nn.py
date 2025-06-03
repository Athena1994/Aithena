

import unittest

import torch

from aithena.nn.dynamic_nn import DynamicNNConfig


class TestDynamicNN(unittest.TestCase):

    def setUp(self):
        self.nn = DynamicNNConfig(**{
            "output-tag": "classifier",

            "units": [
                {
                    "name": "LSTM",
                    "input-tags": ["ts"],
                    "nn": {
                        "type": "LSTM",
                        "options": {
                            "layer-num": 1,
                            "hidden-size": 128
                        }
                    }
                },
                {
                    "name": "fc",
                    "input-tags": ["meta", "LSTM"],
                    "nn": {
                        "type": "Sequential",
                        "options": {
                            "layers": [
                                {"type": "ReLU"},
                                {"type": "Dropout", "options": {"p": 0.5}},
                                {"type": "Linear", "options": {"size": 256}}
                            ]
                        }
                    }
                },
                {
                    "name": "classifier",
                    "input-tags": ["fc"],
                    "nn": {
                        "type": "Sequential",
                        "options": {
                            "layers": [
                                {"type": "ReLU"},
                                {"type": "Dropout", "options": {"p": 0.5}},
                                {"type": "Linear", "options": {"size": 128}},
                                {"type": "ReLU"},
                                {"type": "Dropout", "options": {"p": 0.5}},
                                {"type": "Linear", "options": {"size": 3}}
                            ]
                        }
                    }
                }
            ],
        }).create_network({
            'ts': 6,  # 6 features
            'meta': 2  # 2 features
        }, False)

    def test_init(self):

        self.assertEqual(len(self.nn._units), 3)

        # --- lstm

        self.assertEqual(self.nn._units[0].name, 'LSTM')
        self.assertEqual(self.nn._units[0].concat, False)
        self.assertTrue(isinstance(self.nn._units[0].module,
                                   torch.nn.LSTM))
        self.assertEqual(self.nn._units[0].module.input_size, 6)
        self.assertEqual(self.nn._units[0].module.hidden_size, 128)

        # --- fc

        self.assertEqual(self.nn._units[1].name, 'fc')
        self.assertEqual(self.nn._units[1].concat, True)
        self.assertTrue(isinstance(self.nn._units[1].module,
                                   torch.nn.Sequential))
        self.assertTrue(isinstance(self.nn._units[1].module[0],
                                   torch.nn.ReLU))
        self.assertTrue(isinstance(self.nn._units[1].module[1],
                                   torch.nn.Dropout))
        self.assertEqual(self.nn._units[1].module[1].p, 0.5)
        self.assertTrue(isinstance(self.nn._units[1].module[2],
                                   torch.nn.Linear))
        self.assertEqual(self.nn._units[1].module[2].in_features, 130)
        self.assertEqual(self.nn._units[1].module[2].out_features, 256)

        # --- classifier

        self.assertEqual(self.nn._units[2].name, 'classifier')
        self.assertEqual(self.nn._units[2].concat, False)
        self.assertTrue(isinstance(self.nn._units[2].module,
                                   torch.nn.Sequential))

        self.assertTrue(isinstance(self.nn._units[2].module[0], torch.nn.ReLU))

        self.assertTrue(isinstance(self.nn._units[2].module[1],
                                   torch.nn.Dropout))
        self.assertEqual(self.nn._units[2].module[1].p, 0.5)

        self.assertTrue(isinstance(self.nn._units[2].module[2],
                                   torch.nn.Linear))
        self.assertEqual(self.nn._units[2].module[2].in_features, 256)
        self.assertEqual(self.nn._units[2].module[2].out_features, 128)

        self.assertTrue(isinstance(self.nn._units[2].module[3], torch.nn.ReLU))

        self.assertTrue(isinstance(self.nn._units[2].module[4],
                                   torch.nn.Dropout))
        self.assertEqual(self.nn._units[2].module[4].p, 0.5)

        self.assertTrue(isinstance(self.nn._units[2].module[5],
                                   torch.nn.Linear))
        self.assertEqual(self.nn._units[2].module[5].in_features, 128)
        self.assertEqual(self.nn._units[2].module[5].out_features, 3)

    def test_forward(self):

        # --- input

        ts = torch.rand(5, 128, 6)
        meta = torch.rand(5, 2)

        # --- forward

        out = self.nn({'ts': ts, 'meta': meta})
        self.assertEqual(out.size(), (5, 3))
