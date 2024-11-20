data_root = 'data/Nuscenes-360'

train_dataloader = dict(batch_size=2,
                        num_workers=4,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        collate_fn=dict(type='default_collate'),
                        dataset=dict(type='PanoDataset',
                                     ann_file='nusc_infos_train_mmengine.pkl',
                                     data_root=data_root,
                                     test_mode=False,
                                     pipeline=[
                                         dict(type='LoadImageFromFile'),
                                         dict(type='Resize',
                                              scale=(3200, 180)),
                                         dict(type='LoadBEV'),
                                         dict(type='PackPanoInputs',
                                              meta_keys=('map_classes',
                                                         'frame_name'))
                                     ]))

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      collate_fn=dict(type='default_collate'),
                      dataset=dict(type='PanoDataset',
                                   ann_file='nusc_infos_val_mmengine.pkl',
                                   data_root=data_root,
                                   test_mode=True,
                                   pipeline=[
                                       dict(type='LoadImageFromFile'),
                                       dict(type='Resize', scale=(3200, 180)),
                                       dict(type='LoadBEV'),
                                       dict(type='PackPanoInputs',
                                            meta_keys=('map_classes',
                                                       'frame_name'))
                                   ]))

test_dataloader = val_dataloader

val_evaluator = dict(type='BEVIouMetric',
                     thresholds=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
test_evaluator = val_evaluator
