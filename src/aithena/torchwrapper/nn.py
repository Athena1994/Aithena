from typing import List, Tuple
from torch import nn

from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config


@config
class LinearLayerConfig:
    size: int

    def create(self, input_size: int) -> Tuple[nn.Linear, int]:
        return nn.Linear(input_size, self.size), self.size


@config
class DropoutLayerConfig:
    p: float

    def create(self, input_size: int) -> Tuple[nn.Dropout, int]:
        return nn.Dropout(self.p), input_size


@config
class LSTMLayerConfig:
    hidden_size: int = Attribute('hidden-size')
    layer_num: int = Attribute('layer-num')
    batch_first: bool = Attribute('batch-first', default=True)

    def create(self, input_size: int) -> Tuple[nn.LSTM, int]:
        return nn.LSTM(input_size, self.hidden_size, self.layer_num,
                       batch_first=self.batch_first), self.hidden_size


@config
class Conv1DLayerConfig:
    kernel_size: int = Attribute('kernel-size')
    out_channels: int = Attribute('out-channels')
    stride: int = Attribute(default=1)
    padding: int = Attribute(default=0)

    def create(self, input_size: int) -> Tuple[nn.LSTM, int]:
        return nn.Conv1d(
            input_size, self.out_channels,
            self.kernel_size, self.stride, self.padding), self.out_channels


@config
class AdaptiveAvgPool1DLayerConfig:
    output_size: int = Attribute('output-size')

    def create(self, input_size: int) -> Tuple[nn.AdaptiveAvgPool1d, int]:
        return nn.AdaptiveAvgPool1d(self.output_size), input_size


class SqueezeModule(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


@config
class LayerDescriptor:
    type: str
    options: dict = Attribute('options', default={})

    def create_layer(self, input_size: int = None) -> Tuple[nn.Module, int]:
        """
        Create a layer based on the type and options provided.
        :param input_size: The size of the input to the layer, required for
                           layers like Linear and LSTM. (default: None)
        :return: An instance of the created layer and the output size of the
                 layer. (None if no output size is unknown)
        """
        layer_type = self.type.upper()
        if layer_type == 'RELU':
            return nn.ReLU(), input_size
        elif layer_type == 'DROPOUT':
            return DropoutLayerConfig(**self.options).create(input_size)
        elif layer_type == 'LINEAR':
            return LinearLayerConfig(**self.options).create(input_size)
        elif layer_type == 'LSTM':
            return LSTMLayerConfig(**self.options).create(input_size)
        elif layer_type == 'SEQUENTIAL':
            return SequentialLayerConfig(**self.options).create(input_size)
        elif layer_type == 'CONV1D':
            return Conv1DLayerConfig(**self.options).create(input_size)
        elif layer_type == 'ADAPTIVEAVGPOOL1D':
            return AdaptiveAvgPool1DLayerConfig(**self.options)\
                .create(input_size)
        elif layer_type == 'FLATTEN':
            return nn.Flatten(), input_size
        elif layer_type == 'SQUEEZE':
            return SqueezeModule(), input_size
        else:
            raise ValueError(f"Unknown layer type: {self.type}")


@config
class SequentialLayerConfig:
    layers: List[LayerDescriptor]

    def create(self, input_size: int) -> Tuple[nn.Sequential, int]:

        layers = []

        layer_input = input_size

        for layer in self.layers:
            layer_instance, layer_input = layer.create_layer(layer_input)
            layers.append(layer_instance)

        return nn.Sequential(*layers), layer_input


def initialize_params(module: nn.Module):
    """
    Initialize the parameters of the module.
    :param module: The module to initialize.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.uniform_(m.bias.data)
        if isinstance(m, nn.LSTM):
            nn.init.xavier_normal_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.uniform_(m.bias.data)
        elif isinstance(m, nn.Sequential):
            for submodule in m:
                initialize_params(submodule)
