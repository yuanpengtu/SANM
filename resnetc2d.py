"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class DiversificationBlock(nn.Module):

    def __init__(self, pk=0.5, r=3, c=4):
        super(DiversificationBlock, self).__init__()
        self.pk = pk
        self.r = r
        self.c = c

    def forward(self, feature_maps):
        def helperb1(feature_map):
            row, col = torch.where(feature_map == torch.max(feature_map))
            b1 = torch.zeros_like(feature_map)
            for i in range(len(row)):
                r, c = int(row[i]), int(col[i])
                b1[r, c] = 1
            return b1

        def from_num_to_block(mat, r, c, num):
            assert len(mat.shape) == 2, ValueError("Feature map shape is wrong!")
            res = np.zeros_like(mat)
            row, col = mat.shape
            block_r, block_c = int(row / r), int(col / c)
            index = np.arange(r * c) + 1
            index = index.reshape(r, c)
            index_r, index_c = np.argwhere(index == num)[0]
            if index_c + 1 == c:
                end_c = c + 1
            else:
                end_c = (index_c + 1) * block_c
            if index_r + 1 == r:
                end_r = r + 1
            else:
                end_r = (index_r + 1) * block_r
            res[index_r * block_r: end_r, index_c * block_c:end_c] = 1
            return res

        if len(feature_maps.shape) == 3:
            resb1 = []
            resb2 = []
            feature_maps_list = torch.split(feature_maps, 1)
            for feature_map in feature_maps_list:
                feature_map = feature_map.squeeze()
                tmp = helperb1(feature_map)
                resb1.append(tmp)
                tmp1 = from_num_to_block(feature_map, self.r, self.c, 3)
                resb2.append(tmp1)

        elif len(feature_maps.shape) == 2:
            tmp = helperb1(feature_maps)
            tmp1 = from_num_to_block(feature_maps, self.r, self.c, 3)
            resb1 = [tmp]
            resb2 = [tmp1]

        else:
            raise ValueError
        res = [np.clip(resb1[x].numpy() + resb2[x], 0, 1) for x in range(len(resb1))]
        return res
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False, pool=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if pool:
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if pool else nn.Identity()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropblock = DiversificationBlock()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        featmap = out
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out, featmap


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet50', head='mlp', feat_dim=128, pool=False):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(pool=pool)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        featmap, feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
class Decoder(nn.Module):
    def __init__(self, name='resnet18'):
        super(Decoder, self).__init__()
        if name=='resnet18':
            size = 32
            self.layer1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.layer2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=6, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.layer3 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        else:
            self.layer1 = nn.ConvTranspose2d(in_channels=2048, out_channels=256, kernel_size=6, stride=3, padding=0, output_padding=1, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.layer2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=3, padding=0, output_padding=1, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            self.layer3 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=2, stride=3, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
               
    def forward(self, featmap):
        featmap = self.layer1(featmap)
        featmap = self.layer2(featmap)
        featmap = self.layer3(featmap)
        return featmap

class SupCEResNet(nn.Module):
    """encoder + classifier"""

    def __init__(self, name='resnet50', num_classes=10, pool=False):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(pool=pool)
        self.fc = nn.Linear(dim_in, num_classes)
        self.name = name
        self.recon_module = Decoder(name=name)
    def forward(self, x, return_map=False, recon=False):
        feat, featmap=self.encoder(x)
        score=self.fc(feat)
        if recon:
            recon_img = self.recon_module(featmap)
            return recon_img
        if self.name=='resnet50':
            if return_map:
                return featmap, score
            else:
                return score
        score = F.softmax(score, dim=-1)
        return featmap, feat, score


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
