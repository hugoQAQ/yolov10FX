from ultralytics import YOLOv10
import numpy as np
import torch
import gc

# Correcting the deprecated usage of np.bool
np.bool = np.bool_

# Model configuration
id = "bdd"
model_type = "v10s"
batch_size = 1  # Reduce if memory issues persist
img_size = 800  # Reduce if necessary
ood_datasets = ["ID-voc-OOD-coco", "OOD-open"] if id == "voc" else ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]

# FX
# Load the model once
model = YOLOv10(f"/home/hugo/yolov10/models/{model_type}_{id}/best.engine", task="detect")

# Function to perform validation and clear memory
def validate_and_clear(data, split, project, name):
    model.val(
        data=data,
        batch=batch_size,
        imgsz=img_size,
        verbose=False,
        device="cuda",
        split=split,
        save_json=True,
        project=project,
        name=name,
        conf=0.25
    )
    torch.cuda.empty_cache()
    gc.collect()

# Validation on different splits
# for split in ["train", "val"]:
#     validate_and_clear(
#         data=f"/home/hugo/datasets/{id}/dataset.yaml",
#         split=split,
#         project=f"/home/hugo/yolov10FX/feats/{model_type}_{id}",
#         name=f"{id}-{split}"
#     )

# Validation on out-of-domain datasets
for dataset_name in ood_datasets:
    validate_and_clear(
        data=f"/home/hugo/datasets/{dataset_name}/dataset.yaml",
        project=f"/home/hugo/yolov10FX/feats/{model_type}_{id}",
        name=f"{dataset_name}",
        split="val"
    )
