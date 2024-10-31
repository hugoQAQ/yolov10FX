from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results


class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            nc = preds['one2many'][0].shape[-2] - 4
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.shape[-1] == 6:
            pass
        else:
            if preds.shape[-1] > preds.shape[-2]:
                preds = preds.transpose(-1, -2) # 1, 24, 9975 -> 1, 9975, 24
                bboxes, scores, labels, logits, feats = ops.v10postprocess(preds, self.args.max_det, nc)
                bboxes = ops.xywh2xyxy(bboxes)
                preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1), logits, feats], dim=-1)
            else:
                pass
            

        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], logits=pred[:, 6:nc+6], feats=pred[:, nc+6:]))
        return results
