import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

@METRICS.register_module()
class BEVIouMetric(BaseMetric):
    def __init__(self, thresholds):
        super().__init__(collect_device='cpu')
        self.thresholds = torch.tensor(thresholds)
        self.num_thresholds = len(thresholds)

    def process(self, data_batch, data_samples):
        self.map_classes = data_samples[0]['map_classes']
        self.num_classes = len(self.map_classes)
        
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg'].squeeze()
            label = data_sample['gt_sem_seg'].squeeze().to(pred_label.device)

            pred_label = pred_label.detach().reshape(self.num_classes, -1)
            label = label.detach().bool().reshape(self.num_classes, -1)

            pred_label = pred_label[:, :, None] >= self.thresholds.to(pred_label.device)
            label = label[:, :, None]

            tp = (pred_label & label).sum(dim=1).float()
            fp = (pred_label & ~label).sum(dim=1).float()
            fn = (~pred_label & label).sum(dim=1).float()
            
            result = dict(tp=tp, fp=fp, fn=fn)
            self.results.append(result)
        
    def compute_metrics(self, results):
        tp = torch.zeros(self.num_classes, self.num_thresholds, dtype=torch.float32)
        fp = torch.zeros(self.num_classes, self.num_thresholds, dtype=torch.float32)
        fn = torch.zeros(self.num_classes, self.num_thresholds, dtype=torch.float32)
        
        for result in results:
            tp += result['tp']
            fp += result['fp']
            fn += result['fn']
        ious = tp / (tp + fp + fn + 1e-7)
        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"{name}/iou@max"] = ious[index].max().item()
            # for threshold, iou in zip(self.thresholds, ious[index]):
            #     metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics
