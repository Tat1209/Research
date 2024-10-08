'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import modelutils

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.GroupNorm(groups, planes, affine=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.GroupNorm(groups, planes, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.GroupNorm(groups, self.expansion*planes, affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, groups=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, groups=groups)
        self.bn1 = nn.GroupNorm(groups, planes, affine=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, groups=groups)
        self.bn2 = nn.GroupNorm(groups, planes, affine=False)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False, groups=groups)
        self.bn3 = nn.GroupNorm(groups, self.expansion*planes, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.GroupNorm(groups, self.expansion*planes, affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, nb_fils=64, T=1, groups=1):
        super(ResNet, self).__init__()
        self.in_planes = nb_fils
        self.nb_fils = nb_fils
        self.T = T
        self.groups=groups

        self.conv1 = nn.Conv2d(3*groups, nb_fils, kernel_size=3,
                               stride=1, padding=1, bias=False, groups=groups)
        self.bn1 = nn.GroupNorm(groups, nb_fils, affine=False)
        self.layer1 = self._make_layer(block, nb_fils, num_blocks[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(block, nb_fils*2, num_blocks[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, nb_fils*4, num_blocks[2], stride=2, groups=groups)
        self.layer4 = self._make_layer(block, nb_fils*8, num_blocks[3], stride=2, groups=groups)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(nb_fils*8*block.expansion, num_classes)
        modelutils.initialize_weights(self)

    def _make_layer(self, block, planes, num_blocks, stride, groups=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, groups=groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.groups > 1:
            x = x.repeat(1, self.groups, 1, 1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out /=self.T
        return out

def ResNet18(num=10, nb_fils=64, T=1, groups=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num, nb_fils=nb_fils, T=T, groups=groups)

def ResNet34(num=10, nb_fils=64, T=1, groups=1):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num, nb_fils=nb_fils, T=T, groups=groups)

def ResNet50(num=10, nb_fils=64, T=1, groups=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num, nb_fils=nb_fils, T=T, groups=groups)

def ResNet101(num=10, nb_fils=64, T=1, groups=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num, nb_fils=nb_fils, T=T, groups=groups)

def ResNet152(num=10, nb_fils=64, T=1, groups=1):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num, nb_fils=nb_fils, T=T, groups=groups)

