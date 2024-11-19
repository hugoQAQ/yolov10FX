from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results


class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            nc = preds['one2many'][0].shape[-2] - 4
            one2one_preds = preds["one2one"]
            cv3_feats = preds['cv3_features']
        else:
            one2one_preds = preds
            cv3_feats = None

        if isinstance(one2one_preds, (list, tuple)):
            preds = one2one_preds[0]


        if preds.shape[-1] == 6:
            pass
        else:
            if preds.shape[-1] > preds.shape[-2]:
                preds = preds.transpose(-1, -2)
                bboxes, scores, labels, logits, cv3_branch1_cat, cv3_branch2_cat, cv3_mask_cat = ops.v10postprocess(preds, cv3_feats, self.args.max_det, nc)
                bboxes = ops.xywh2xyxy(bboxes)
                preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1), logits], dim=-1)
            else:
                pass
            

        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]
        cv3_branch1_cat = [p[mask[idx]] for idx, p in enumerate(cv3_branch1_cat)]
        cv3_branch2_cat = [p[mask[idx]] for idx, p in enumerate(cv3_branch2_cat)]
        cv3_mask_cat = [p[mask[idx]] for idx, p in enumerate(cv3_mask_cat)]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            logits = pred[:, 6:]
            branch1_feats = cv3_branch1_cat[i]
            branch2_feats = cv3_branch2_cat[i]
            mask_feats = cv3_mask_cat[i].cpu().numpy().flatten().tolist()
            results.append(Results(orig_img, 
                                 path=img_path, 
                                 names=self.model.names, 
                                 boxes=pred[:, :6], 
                                 logits=logits, 
                                 feats={"branch1_outputs": branch1_feats, 
                                       "branch2_outputs": branch2_feats,
                                       "head_idx": mask_feats}))
        return results