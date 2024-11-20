import torch
from torch import nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmengine.registry import MODELS

@MODELS.register_module()
class NeckOneBEV(BaseModule):

    def __init__(self, in_channels=[96, 192, 384, 768], out_channels=128, decoder_output_dim=256):
        super().__init__()
        self.linear_c4 = nn.Linear(in_channels[3], decoder_output_dim)
        self.linear_c3 = nn.Linear(in_channels[2], decoder_output_dim)
        self.linear_c2 = nn.Linear(in_channels[1], decoder_output_dim)
        self.linear_fuse = nn.Conv2d(3 * decoder_output_dim, out_channels, 1)

    def forward(self, feature_maps):
        batch_size, _, _, _ = feature_maps[0].shape
        dtype = feature_maps[0].dtype

        c4 = feature_maps[3]
        c3 = feature_maps[2]
        c2 = feature_maps[1]

        # Flatten and transpose the feature maps
        c4_flattened = c4.flatten(2).transpose(1, 2).contiguous()
        c3_flattened = c3.flatten(2).transpose(1, 2).contiguous()
        c2_flattened = c2.flatten(2).transpose(1, 2).contiguous()

        # Apply linear layers
        c4_transformed = self.linear_c4(c4_flattened).permute(0, 2, 1).reshape(batch_size, -1, c4.shape[2], c4.shape[3]).contiguous()
        c3_transformed = self.linear_c3(c3_flattened).permute(0, 2, 1).reshape(batch_size, -1, c3.shape[2], c3.shape[3]).contiguous()
        c2_transformed = self.linear_c2(c2_flattened).permute(0, 2, 1).reshape(batch_size, -1, c2.shape[2], c2.shape[3]).contiguous()

        # Interpolate the transformed feature maps to match the size of c2
        c4_resized = F.interpolate(c4_transformed, size=c2.size()[2:], mode='bilinear', align_corners=False)
        c3_resized = F.interpolate(c3_transformed, size=c2.size()[2:], mode='bilinear', align_corners=False)
        c2_resized = F.interpolate(c2_transformed, size=c2.size()[2:], mode='bilinear', align_corners=False)

        # Fuse the feature maps
        fused_feature_map = self.linear_fuse(torch.cat([c4_resized, c3_resized, c2_resized], dim=1))

        return [fused_feature_map]
