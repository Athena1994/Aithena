import copy
from typing import List
import torch.nn as nn

from aithena.nn.nn_unit import NNUnit, UnitDescriptor
from aithena.torchwrapper.nn import initialize_params

from jodisutils.config.attribute import Attribute
from jodisutils.config.decorator import config


@config
class DynamicNNConfig:
    units: List[UnitDescriptor]
    output_tag: str = Attribute('output-tag')

    def create_network(self, tagged_input_sizes: dict, cuda: bool) \
            -> 'DynamicNN':
        nn = DynamicNN(self.output_tag, self.units, tagged_input_sizes)
        if cuda:
            nn = nn.cuda()
        return nn


class DynamicNN(nn.Module):

    def __init__(self,
                 output_tag: str,
                 unit_descriptors: List[UnitDescriptor],
                 tagged_input_sizes: dict):
        super(DynamicNN, self).__init__()

        self._units: List[NNUnit] = []
        self._output_tag = output_tag

        tagged_output_sizes = copy.deepcopy(tagged_input_sizes)

        for unit_desc in unit_descriptors:
            unit, tagged_output_sizes[unit_desc.name] \
                  = unit_desc.create_unit(tagged_output_sizes)
            self._units.append(unit)
            self.add_module(unit.name, unit.module)

        # assert that the output key is present in the generated output
        if output_tag not in tagged_output_sizes:
            raise ValueError(f"Output key {output_tag} not found")

        initialize_params(self)

    @property
    def output_tag(self) -> str:
        return self._output_tag

    def forward(self, x_dict):
        for unit in self._units:
            unit.forward(x_dict)

        return x_dict[self.output_tag]
