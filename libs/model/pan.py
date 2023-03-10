import torch
from torch import nn
from torch.nn import functional as F


class PAN(nn.Module):
    def __init__(self, num_levels, in_channels, out_channels):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pan_layers = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.pan_layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
            )

    def forward(self, feats):
        p2, p3, p4, p5 = feats
        p2_ = p2
        p3_ = self.pan_layers[0](F.interpolate(p2_, size=p3.shape[2:], align_corners=False, mode='bilinear') + p3)
        p4_ = self.pan_layers[1](F.interpolate(p3_, size=p4.shape[2:], align_corners=False, mode='bilinear') + p4)
        p5_ = self.pan_layers[2](F.interpolate(p4_, size=p5.shape[2:], align_corners=False, mode='bilinear') + p5)
        return p5_
