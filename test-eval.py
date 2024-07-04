from ultralytics import YOLOv10
import numpy as np
CUDA_LAUNCH_BLOCKING=1
np.bool = np.bool_
model = YOLOv10("/home/hugo/yolov10/models/v10s_voc/weights/best.engine", task="detect")
results = model.val(
    data="/home/hugo/datasets/voc/dataset.yaml", 
    batch=8,
    imgsz=800,
    verbose=False,
    device="cuda",
    split="val", 
    save_json=True,
    conf=0.25
)