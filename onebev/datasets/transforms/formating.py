import numpy as np
from mmcv import BaseTransform
from mmengine.registry import TRANSFORMS
from ...structures import SegDataSample
from mmcv.transforms import to_tensor


@TRANSFORMS.register_module()
class PackPanoInputs(BaseTransform):

    def __init__(self, meta_keys='map_classes'):
        self.meta_keys = meta_keys

    def transform(self, results):
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()

        if 'gt_masks_bev' in results:
            data_sample.gt_sem_seg = to_tensor(results['gt_masks_bev'])

        if self.meta_keys is not None:
            img_meta = {}
            for key in self.meta_keys:
                if key in results:
                    img_meta[key] = results[key]
            data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results
