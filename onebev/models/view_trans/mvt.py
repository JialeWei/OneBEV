import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule, ModuleList, xavier_init
from mmengine.registry import MODELS


@MODELS.register_module()
class Sampling(BaseModule):

    def __init__(self, embed_dims=128, num_locations=5, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_levels = 1
        self.num_points = num_locations * num_locations
        self.num_locations = num_locations
        self.sampling_points = nn.Linear(embed_dims, num_locations * 2)
        self.sampling_offsets = nn.Linear(embed_dims, self.num_points * 2)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.sampling_offsets, distribution='uniform', bias=0.)
        xavier_init(self.sampling_points, distribution='uniform', bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)

    @staticmethod
    def sampling(
        value,
        value_spatial_shapes,
        sampling_locations,
        bev_h,
        bev_w,
    ):
        bs, _, embed_dims = value.shape
        _, _, _, num_points, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                                 dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (H_, W_) in enumerate(value_spatial_shapes):
            # bs, H_*W_, embed_dims ->
            # bs, embed_dims, H_, W_
            value_l_ = value_list[level].transpose(1, 2).reshape(
                bs, embed_dims, H_, W_)
            # bs, num_queries, num_points, 2 ->
            sampling_grid_l_ = sampling_grids[:, :, level]
            # bs, embed_dims, num_queries, num_points
            sampling_value_l_ = F.grid_sample(value_l_,
                                              sampling_grid_l_,
                                              mode='bilinear',
                                              padding_mode='zeros',
                                              align_corners=False)
            sampling_value_list.append(sampling_value_l_)

        output = torch.stack(sampling_value_list, dim=-2).flatten(-2)

        output = output.view(bs, embed_dims, -1, int(math.sqrt(num_points)),
                             int(math.sqrt(num_points)))
        output = output.view(bs, embed_dims, bev_h, bev_w,
                             int(math.sqrt(num_points)),
                             int(math.sqrt(num_points)))
        output = output.permute(0, 1, 2, 4, 3,
                                5).reshape(bs, embed_dims,
                                           bev_h * int(math.sqrt(num_points)),
                                           bev_w * int(math.sqrt(num_points)))
        return output

    def forward(self, query, value, bev_pos, spatial_shapes, bev_h, bev_w,
                **kwargs):
        bs, num_query, _ = query.shape 
        _, num_value, bs, _ = value.shape  
        value = value.permute(2, 0, 1, 3).reshape(
            bs, num_value, self.embed_dims
        )  
        reference_points = self.sampling_points(bev_pos).sigmoid().view(
            bs, num_query, self.num_locations,
            2)  
        value = self.value_proj(value).view(
            bs, num_value,
            self.embed_dims) 
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_levels, self.num_points,
            2)  
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]],
            -1) 

        bs, num_query, num_locations, xy = reference_points.shape  

        reference_points = reference_points[:, :, None,
                                            None, :, :]  
        sampling_offsets = sampling_offsets / \
            offset_normalizer[None, None, :, None, :] 
        bs, num_query, num_levels, num_all_points, xy = sampling_offsets.shape  
        sampling_offsets = sampling_offsets.view(
            bs, num_query, num_levels, num_all_points // num_locations,
            num_locations, xy
        )  
        sampling_locations = reference_points + sampling_offsets  
        bs, num_query, num_levels, num_points, num_locations, xy = sampling_locations.shape
        assert num_all_points == num_points * num_locations

        sampling_locations = sampling_locations.view(
            bs, num_query, num_levels, num_all_points,
            xy)  
        output = self.sampling(value, spatial_shapes, sampling_locations,
                               bev_h, bev_w)
        return output


@MODELS.register_module()
class MVTLayer(BaseModule):

    def __init__(self,
                 sampling_cfgs=None,
                 vssm_cfgs=None,
                 **kwargs):

        super().__init__()
        self.vssm = MODELS.build(vssm_cfgs)
        self.sampling = MODELS.build(sampling_cfgs)

    def forward(self,
                query,
                value=None,
                bev_pos=None,
                bev_h=None,
                bev_w=None,
                spatial_shapes=None,
                **kwargs):
        query = self.sampling(query, value, bev_pos, spatial_shapes, bev_h,
                              bev_w, **kwargs)
        query = self.vssm(query)
        return query


@MODELS.register_module()
class MambaViewTransformation(BaseModule):

    def __init__(self, num_layers, mvt_layers=None):
        super().__init__()

        if isinstance(mvt_layers, dict):
            mvt_layers = [copy.deepcopy(mvt_layers) for _ in range(num_layers)]
        else:
            assert isinstance(mvt_layers, list) and \
                len(mvt_layers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(MODELS.build(mvt_layers[i]))

    def forward(self,
                bev_query,
                value,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        bev_query = bev_query.permute(
            1, 0, 2)  
        bev_pos = bev_pos.permute(
            1, 0, 2)  
        for layer in self.layers:
            output = layer(
                bev_query, 
                value,  
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos, 
                spatial_shapes=spatial_shapes,
                **kwargs)
            bev_query = output
        return output
