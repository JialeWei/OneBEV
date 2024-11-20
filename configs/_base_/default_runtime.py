load_from = None
resume = False
launcher = 'none'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
visualizer = dict(type='Visualizer')

model_wrapper_cfg=dict(
    type='MMDistributedDataParallel', find_unused_parameters=True)
