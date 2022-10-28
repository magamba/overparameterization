from enum import Enum
import abc

import torch
import torch.nn as nn


class NetworkAddition(Enum):
    BATCH_NORM = "batch_norm"
    DROPOUT = "dropout"


ALL_ADDITIONS = {NetworkAddition.BATCH_NORM.value, NetworkAddition.DROPOUT.value}


class NetworkBuilder(abc.ABC):
    def __init__(self, dataset_info):
        self._dataset_info = dataset_info

    @abc.abstractmethod
    def add(self, addition: NetworkAddition, **kwargs):
        """Add the network component addition, to the network"""

    @abc.abstractmethod
    def build_net(self) -> nn.Module:
        """
        Take whatever internal state this keeps and convert it into a module
        object to be consumed metrics
        """

