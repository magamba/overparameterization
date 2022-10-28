import torch
import torch.nn as nn
from numpy import prod

from core.data import DatasetInfos
from models.concepts import NetworkBuilder, NetworkAddition
from functools import partial

""" mCNN architecture from Nakkiran et al. 2019
"""

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

class ConvBlock(nn.Module):
    """
    Convolutional block with optional batch normalization, in the form
    Conv -> (Batch Norm) -> ReLU
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        use_batch_norm=False,
    ):
        super(ConvBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("ConvBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = conv3x3(inplanes, planes, stride, bias=not use_batch_norm)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.use_batch_norm = use_batch_norm
        if not use_batch_norm: 
            # disable batch norm at this stage to ensure reproducibility with prev
            # versions of the code, which always initialized batch norm layers
            self.bn = None

    def forward(self, x):

        out = self.conv(x)
        if self.use_batch_norm:
            out = self.bn(out)
        out = self.relu(out)

        return out


class mCNN(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        groups=1,
        width_per_group=64,
        use_batch_norm=False,
        in_channels=3,
        inplanes=64,
        channels=(128, 256, 512),
        weight_init="pytorch"
    ):
        super(mCNN, self).__init__()

        self.inplanes = inplanes
        self.dilation = 1
        self._use_batch_norm = use_batch_norm
        self.groups = groups
        self.base_width = width_per_group
        
        self.pool = nn.AvgPool2d(2)
        self.apool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], in_channels=in_channels)
        self.layer2 = self._make_layer(
            block,
            channels[0],
            layers[1],
        )
        self.layer3 = self._make_layer(
            block,
            channels[1],
            layers[2],
        )
        self.layer4 = self._make_layer(
            block,
            channels[2],
            layers[3],
        )
        self.fc = nn.Linear(channels[2] * block.expansion, num_classes)
        
        for m in self.modules():
            if weight_init == "nkaiming":
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode='fan_out',
                        nonlinearity= 'linear' if isinstance(m, nn.Linear) else 'relu'
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, in_channels=None):
        if in_channels is not None:
            inplanes = in_channels
        else:
            inplanes = self.inplanes
        layers = [
            block(
                inplanes,
                planes,
                stride,
                self.groups,
                self.base_width,
                use_batch_norm=self.use_batch_norm,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    use_batch_norm=self.use_batch_norm,
                )
            )
        if blocks > 1:
            return nn.Sequential(*layers)
        return layers[0]

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.apool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def use_batch_norm(self):
        return self._use_batch_norm

    def _set_use_batch_norm_layer(self, seq_layer):
        for block in seq_layer:
            block.use_batch_norm = self.use_batch_norm

    @use_batch_norm.setter
    def use_batch_norm(self, truth_value):
        self._use_batch_norm = truth_value
        self._set_use_batch_norm_layer(self.layer1)
        self._set_use_batch_norm_layer(self.layer2)
        self._set_use_batch_norm_layer(self.layer3)
        self._set_use_batch_norm_layer(self.layer4)


class CNNBuilder(NetworkBuilder):
    def __init__(self, cnn_cls, block_cls, arch, dataset_info, **kwargs):
        self._model = cnn_cls(
            block_cls,
            arch,
            num_classes=dataset_info.output_dimension,
            in_channels=dataset_info.input_shape[0],
            **kwargs,
        )
        super().__init__(dataset_info)

    def add(self, addition: NetworkAddition, **kwargs):
        if addition == NetworkAddition.BATCH_NORM:
            self.add_batch_norm()
        if addition == NetworkAddition.DROPOUT:
            self.add_dropout()

    def add_batch_norm(self):
        self._model.use_batch_norm = True

    def add_dropout(self):
        raise NotImplementedError("Dropout not supported yet")

    def build_net(self) -> nn.Module:
        return self._model


# ConvNets of increasing base width, following Nakkiran et al. 2019
cnn5_increasing_widths = {
    "cnn5_" + str(width): partial(CNNBuilder, mCNN, ConvBlock, [1, 1, 1, 1], inplanes=width, channels=(2*width, 4*width, 8*width)) for width in range(1,65)
}


MODEL_FACTORY_MAP = {
    **cnn5_increasing_widths,
}
