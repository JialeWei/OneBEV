import h5py
from mmcv import BaseTransform
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadBEV(BaseTransform):
    def transform(self, data):
        bev_path = data["bev_path"]
        with h5py.File(bev_path, 'r') as hf:
            labels = hf['labels'][:]
        data["gt_masks_bev"] = labels
        return data
