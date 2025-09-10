import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from .sparsebev_transformer import SparseBEVSelfAttention, SparseBEVSampling, AdaptiveMixing
from .utils import DUMP, generate_grid, batch_indexing
from .bbox.utils import encode_bbox


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 conv_cfg=dict(type='Conv3d'), 
                 norm_cfg=dict(type='BN3d'), 
                 act_cfg=dict(type='ReLU',inplace=True)):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvModule(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, 
            act_cfg=None,
        )
        self.downsample = ConvModule(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0,
            conv_cfg=conv_cfg, 
            norm_cfg=norm_cfg, 
            act_cfg=None,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.downsample(x)
        out = F.relu(out)
        return out

class TemporalNet(nn.Module):
    def __init__(self, in_channels=240,
                 out_channels=[16, 128, 64, 32],
                 conv_cfg=dict(type='Conv3d'), 
                 norm_cfg=dict(type='BN3d'), 
                 act_cfg=dict(type='ReLU',inplace=True),
                 ):
        super(TemporalNet, self).__init__()
        self.conv_head = ConvModule(in_channels, 
                                    out_channels[0],
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1, 
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg, 
                                    act_cfg=act_cfg)

        self.layer1 = self.make_layer(out_channels[0], out_channels[1], num_blocks=2, 
                                      conv_cfg=conv_cfg, 
                                      norm_cfg=norm_cfg, 
                                      act_cfg=act_cfg)

        self.layer2 = self.make_layer(out_channels[1], out_channels[2], num_blocks=2, 
                                      conv_cfg=conv_cfg, 
                                      norm_cfg=norm_cfg, 
                                      act_cfg=act_cfg)

        self.layer3 = self.make_layer(out_channels[2], out_channels[3], num_blocks=2, 
                                      conv_cfg=conv_cfg, 
                                      norm_cfg=norm_cfg, 
                                      act_cfg=act_cfg)

        self.conv_back = ConvModule(out_channels[3], 
                                    2,
                                    kernel_size=1, 
                                    stride=1, 
                                    padding=0, 
                                    conv_cfg=conv_cfg, 
                                    norm_cfg=norm_cfg, 
                                    act_cfg=None)

    def make_layer(self, in_channels, 
                        out_channels, 
                        num_blocks=2, 
                        conv_cfg=dict(type='Conv3d'), 
                        norm_cfg=dict(type='BN3d'), 
                        act_cfg=dict(type='ReLU',inplace=True)):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, 
                                        out_channels, 
                                        conv_cfg=conv_cfg, 
                                        norm_cfg=norm_cfg, 
                                        act_cfg=act_cfg))
            in_channels = out_channels # after one round, the inchannel will become outchannel
        return nn.Sequential(*layers)

    def forward(self, bev_3d):
        bev_3d = self.conv_head(bev_3d)
        bev_3d = self.layer1(bev_3d)
        bev_3d = self.layer2(bev_3d)
        bev_3d = self.layer3(bev_3d)
        bev_3d = self.conv_back(bev_3d)
        return bev_3d

