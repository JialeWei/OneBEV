_base_ = ['../../_base_/models/model_onebev.py',
          '../../_base_/datasets/deep_dataset.py',
          '../../_base_/schedules/schedule_deep.py',
          '../../_base_/default_runtime.py']

model = dict(head=dict(num_map_classes=17))