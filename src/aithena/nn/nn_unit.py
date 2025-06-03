

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Union

import torch

from aithena.torchwrapper.nn import LayerDescriptor

from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config


@config
class UnitDescriptor:
    name: str
    nn: LayerDescriptor
    input_tags: List[str] = Attribute('input-tags')

    def create_unit(self, input_sizes: Dict[str, Union[tuple, int]]) \
            -> Tuple['NNUnit', int]:

        def concatenate_shapes(sizes: Iterable[Union[tuple, int]]) -> tuple:
            # convert all sizes to tuples if they are not already
            shapes = [s if isinstance(s, tuple) else (s,) for s in sizes]
            # find the maximum dimension of all shapes
            max_dim = max([len(s) for s in shapes])
            # give all shapes same number of dimensions
            shapes = [s + (1,) * (max_dim - len(s)) for s in shapes]
            # assert only the last dimension is different
            if any(s[:-1] != shapes[0][:-1] for s in shapes):
                raise ValueError("All input sizes must have the same shape "
                                 "except for the last dimension.")
            # sum the last dimensions of all shapes
            if len(shapes[0]) == 1:
                return (sum(s[0] for s in shapes),)
            else:
                return (*shapes[0][:-1], sum(s[-1] for s in shapes))

        # assert that all input tags are present in the output sizes
        missing = list(filter(lambda n: n not in input_sizes, self.input_tags))
        if any(missing):
            raise ValueError(f"Input tags {missing} missing...")

        # calculate total input size
        input_size = concatenate_shapes(
            [input_sizes[t] for t in self.input_tags])[-1]

        # create the unit module and get output sizes
        unit_module, output_size = self.nn.create_layer(input_size)

        return NNUnit(
            name=self.name,
            concat=len(self.input_tags) > 1,
            module=unit_module,
            input_tags=self.input_tags
        ), output_size


@dataclass
class NNUnit:
    """
    Represents a logical unit of layers in a neural network. They are
    supposed to be used sequentially in a DynamicNN. A unit consumes one or
    multiple tagged input tensors and produces a tagged output tensor. The
    output tag is the same as the unit name.
    """
    name: str
    module: torch.nn.Module
    concat: bool
    input_tags: List[str]

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> None:

        if self.concat:  # concatenate required input tensors
            inputs = [x_dict[i] for i in self.input_tags]
            x = torch.cat(inputs, dim=1)
        else:
            x = x_dict[self.input_tags[0]]

        is_lstm = isinstance(self.module, torch.nn.LSTM)
        if is_lstm:
            self.module.flatten_parameters()

        x = self.module(x)
        if is_lstm:
            self.module.flatten_parameters()
            x = x[0][:, 0, :]

        x_dict[self.name] = x
