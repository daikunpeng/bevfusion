# Copyright 2021 Toyota Research Institute.  All rights reserved.
# Adapted from:
#    https://github.com/ucbdrive/dla/blob/master/dla.py
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

__all__ = ["DLA"]#把名字为 DLA 的类暴露到模块外

class BasicBlock(nn.Module):
    def __init__(
        self, inplanes, planes, stride=1, dilation=1, 
        conv_cfg=dict(type="Conv2d"), norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(BasicBlock, self).__init__()
        # layer one: conv1
        # inplanes：输入通道数。它表示输入特征图的通道数。

        # planes：输出通道数。它表示卷积操作后输出特征图的通道数。

        # kernel_size：卷积核大小。指定卷积核的高度和宽度。

        # stride：卷积步幅。指定卷积操作在水平和垂直方向上的步幅大小。

        # padding：填充大小。指定在输入特征图周围添加零填充的大小。通常用于控制输出特征图的尺寸。

        # bias：一个布尔值，表示是否在卷积中包含偏置项。如果 norm_cfg 为 None，则偏置项会被包含，否则不包含。

        # dilation：卷积扩展率。指定卷积核内元素之间的间隔。用于增加感受野大小。

        # conv_cfg：卷积配置。可能包含卷积的类型、参数等配置信息。

        # norm_cfg：规范化配置。可能包含规范化的类型、参数等配置信息。

        # act_cfg：激活函数配置。可能包含激活函数的类型、参数等配置信息。


        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=norm_cfg is None,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        # layer two: conv2
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=norm_cfg is None,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x) # 输入张量前向传播
        out = F.relu_(out) 

        out = self.conv2(out) # 张量进入 conv2

        # 为什么残差要有 residual 这个参数？因为残差连接传递过去的tensor可能与conv2出来的维度不一样，可能需要转换
        # 但是此例中不需要这个操作
        out = out + residual
        out = F.relu_(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2 # 这样定义了一个类属性，访问它就是用 Bottleneck.expansion

    def __init__(
        self, inplanes, planes, stride=1, dilation=1,
        conv_cfg=dict(type="Conv2d"), norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion #把类属性赋值给本地变量 expansion
        bottle_planes = planes // expansion
        # layer 1: conv1
        self.conv1 = ConvModule(
            inplanes, 
            bottle_planes, 
            kernel_size=1, 
            bias=norm_cfg is None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        # layer 2: conv2
        self.conv2 = ConvModule(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=norm_cfg is None,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        # layer 3: conv3
        self.conv3 = ConvModule(
            bottle_planes, 
            planes, 
            kernel_size=1, 
            bias=norm_cfg is None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.stride = stride
    # 前向传播
    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        out = out + residual
        out = F.relu_(out)

        return out


class Root(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, residual, 
        conv_cfg=dict(type="Conv2d"), norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(Root, self).__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=norm_cfg is None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.residual = residual

    '''
    在你提供的代码中，`*x` 表示一个可变长度的参数列表，也就是说这个函数可以接受任意数量的参数。在函数内部，`x` 将被视为一个元组，其中包含了传递给函数的所有参数。这种语法允许你在不知道函数会接受多少个参数的情况下灵活地定义函数。

    具体来说，这段代码中的 `forward` 方法接受任意数量的输入参数，并将它们存储在 `children` 变量中。然后，它使用 `torch.cat(x, 1)` 将输入参数 `x` 在维度1上拼接起来，这通常用于将多个输入特征图连接在一起。

    这种方式的设计通常用于深度学习中，特别是在处理具有不同分支或多个输入的模型时，可以通过 `*args` 或 `*kwargs` 这样的参数形式来接受不定数量的输入。这种方式可以增加函数的灵活性和通用性。
    '''

    def forward(self, *x):
        children = x
        y = self.conv(torch.cat(x, 1))
        if self.residual:
            y = y + children[0]
        y = F.relu_(y)

        return y


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
        conv_cfg=dict(type="Conv2d"), 
        norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                conv_cfg=conv_cfg, 
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                conv_cfg=conv_cfg, 
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        # (dennis.park) If 'self.tree1' is a Tree (not BasicBlock), then the output of project is not used.
        # if in_channels != out_channels:
        if in_channels != out_channels and not isinstance(self.tree1, Tree):
            self.project = ConvModule(
                in_channels, out_channels, kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        # (dennis.park) If 'self.tree1' is a 'Tree', then 'residual' is not used.
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            y = self.root(x2, x1, *children)
        else:
            children.append(x1)
            y = self.tree2(x1, children=children)
        return y

@BACKBONES.register_module()
class DLA(BaseModule):
    def __init__(
        self,
        levels,
        channels,
        block=BasicBlock,
        residual_root=False,
        norm_eval=False,
        out_features=None,
        conv_cfg=dict(type="Conv2d"), 
        norm_cfg=dict(type="BN2d"),
        act_cfg=None
    ):
        super(DLA, self).__init__()
        self.channels = channels
        self.base_layer = ConvModule(
            3,
            channels[0],
            kernel_size=7,
            stride=1,
            padding=3,
            bias=norm_cfg is None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="ReLU")
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0], conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2, conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level2 = Tree(
            levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level3 = Tree(
            levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level4 = Tree(
            levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.level5 = Tree(
            levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.norm_eval = norm_eval

        if out_features is None:
            out_features = ['level5']
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

        out_feature_channels, out_feature_strides = {}, {}
        for lvl in range(6):
            name = f"level{lvl}"
            out_feature_channels[name] = channels[lvl]
            out_feature_strides[name] = 2**lvl

        self._out_feature_channels = {name: out_feature_channels[name] for name in self._out_features}
        self._out_feature_strides = {name: out_feature_strides[name] for name in self._out_features}

    @property
    def size_divisibility(self):
        return 32

    def _make_conv_level(self, inplanes, planes, convs, conv_cfg, norm_cfg, act_cfg=None, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(
                ConvModule(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=norm_cfg is None,
                    dilation=dilation,
                    conv_cfg=conv_cfg, 
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="ReLU")
                )
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        assert x.dim() == 4, f"DLA takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.base_layer(x)
        for i in range(6):
            name = f"level{i}"
            x = self._modules[name](x)
            if name in self._out_features:
                outputs[name] = x
        return outputs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(DLA, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
