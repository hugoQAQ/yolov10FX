from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
import torch

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Acknowledgement: Thanks to sanha9999 in #190 and #181!
        if preds.shape[-1] == 6:
            return preds
        else:
            if preds.shape[-1] > preds.shape[-2]:
                preds = preds.transpose(-1, -2)
                bboxes, scores, labels, logits = ops.v10postprocess(preds, self.args.max_det, self.nc)
                bboxes = ops.xywh2xyxy(bboxes)
                return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1), logits], dim=-1)
            else:
                return preds