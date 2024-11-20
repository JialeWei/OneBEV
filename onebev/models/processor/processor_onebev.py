import torch
from torch import nn

from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class ProcessorOneBEV(BaseModule):

    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=128,
                 positional_encoding=dict(type='SinePositionalEncoding',
                                          num_feats=64,
                                          normalize=True),
                 mvt=None):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w,
                                          self.embed_dims)
        self.positional_encoding = MODELS.build(positional_encoding)
        self.mvt = MODELS.build(mvt)

    def get_bev_features(self,
                         mlvl_feats,
                         bev_queries,
                         bev_h,
                         bev_w,
                         bev_pos=None,
                         **kwargs):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs,
                                                      1)  
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(0, 2,
                                           1)  
            feat = feat.unsqueeze(0)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten,
                                 2) 
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long,
            device=bev_pos.device) 
        
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1,
                                            3)  

        bev_embed = self.mvt(
            bev_queries,  
            feat_flatten,  
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs)

        return bev_embed

    def forward(self, x):
        mlvl_feats = x
        bs, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        outputs = self.get_bev_features(mlvl_feats, bev_queries, self.bev_h,
                                        self.bev_w, bev_pos)

        outputs = outputs.view(bs, self.bev_h, self.bev_w, self.embed_dims)
        outputs = outputs.permute(0, 3, 1, 2).contiguous()
        return outputs.contiguous()