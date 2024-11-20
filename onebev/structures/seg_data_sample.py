from torch import Tensor
from mmengine.structures import BaseDataElement

class SegDataSample(BaseDataElement):

    @property
    def gt_sem_seg(self) -> Tensor:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: Tensor) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=Tensor)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> Tensor:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: Tensor) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=Tensor)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> Tensor:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: Tensor) -> None:
        self.set_field(value, '_seg_logits', dtype=Tensor)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits