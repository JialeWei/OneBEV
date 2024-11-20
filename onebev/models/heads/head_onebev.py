from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class HeadOneBEV(BaseModule):

    def __init__(self, embed_dims, num_map_classes, loss_type):
        super().__init__()
        self.feat_dim = embed_dims
        self.num_map_classes = num_map_classes
        self.loss_type = MODELS.build(loss_type)

        self.layer = nn.Sequential(
            nn.Conv2d(embed_dims,
                      128,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,
                      num_map_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )

    def forward(self, feats):
        outs = self.layer(feats)
        return outs

    def loss(self, feats, batch_data_samples):
        outs = self.layer(feats)
        losses = self.loss_type(outs, batch_data_samples)
        outputs = {}
        for name, val in losses.items():
            if val.requires_grad:
                outputs[f"loss/map/{name}"] = val * 1.0
            else:
                outputs[f"stats/map/{name}"] = val
        return outputs

    def predict(self, feats):
        outs = self.layer(feats)
        return outs
