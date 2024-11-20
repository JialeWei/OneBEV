from mmengine.registry import DATASETS
from mmengine.dataset import BaseDataset
from mmengine.fileio import load


@DATASETS.register_module()
class PanoDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 data_root,
                 pipeline,
                 data_prefix=dict(img_path=''),
                 indices=None,
                 test_mode=False):
        super().__init__(ann_file=ann_file,
                         data_root=data_root,
                         data_prefix=data_prefix,
                         pipeline=pipeline,
                         indices=indices,
                         test_mode=test_mode)

    def load_data_list(self):
        annotations = load(self.ann_file)
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        data_list = []
        for raw_data_info in raw_data_list:
            data_list.append(
                dict(img_path=raw_data_info['img_path'],
                     frame_name=raw_data_info['frame_name'],
                     bev_path=raw_data_info['bev_path'],
                     map_classes=metainfo['map_classes']))
        return data_list
