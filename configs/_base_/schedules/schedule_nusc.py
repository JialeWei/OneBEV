# optimizer
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW',
                                    lr=0.0005,
                                    betas=[0.9, 0.999],
                                    weight_decay=0.01),
                     clip_grad=dict(max_norm=35, norm_type=2))

# lr scheduler
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.1,
         end_factor=1,
         by_epoch=True,
         begin=0,
         end=22,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR',
         by_epoch=True,
         eta_min=0,
         T_max=28,
         begin=22,
         end=50,
         convert_to_iter_based=True)
]

# train/val/test loop settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# hooks
default_hooks = dict(timer=dict(type='IterTimerHook'),
                     logger=dict(type='LoggerHook', interval=100),
                     param_scheduler=dict(type='ParamSchedulerHook'),
                     checkpoint=dict(type='CheckpointHook',
                                     interval=1,
                                     max_keep_ckpts=1),
                     sampler_seed=dict(type='DistSamplerSeedHook'))



