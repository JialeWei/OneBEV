from ...structures import SegDataSample

from mmseg.models import BaseSegmentor
from mmengine.registry import MODELS

@MODELS.register_module()
class SegOneBEV(BaseSegmentor):
    def __init__(
        self,
        data_preprocessor=None,
        backbone=None,
        neck=None,
        view_trans=None,
        head=None,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.view_trans = MODELS.build(view_trans)
        self.head = MODELS.build(head)

    def extract_feat(self, pano):  
        x = self.backbone(pano)
        if hasattr(self, "neck"):
            x = self.neck(x)
        x = self.view_trans(x)
        return x

    def _forward(self, batch_inputs, batch_data_samples=None):
        feats = self.extract_feat(batch_inputs)
        outs = self.head(feats)
        return outs

    def loss(self, batch_inputs, batch_data_samples=None):
        feats = self.extract_feat(batch_inputs)
        losses = self.head.loss(feats, batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples=None):
        feats = self.extract_feat(batch_inputs)
        seg_preds = self.head.predict(feats)
        return self.postprocess_result(seg_preds, batch_data_samples)

    def encode_decode(self, batch_inputs, batch_data_samples):
        pass

    def postprocess_result(self, seg_logits, data_samples):
        batch_size, _, _, _ = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                i_seg_logits = seg_logits[i].sigmoid()
                i_seg_pred = i_seg_logits
            data_samples[i].set_data({
                'seg_logits': i_seg_logits,
                'pred_sem_seg': i_seg_pred
            })
        return data_samples
