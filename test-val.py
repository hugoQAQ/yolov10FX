from ultralytics import YOLOv10
model = YOLOv10('/home/hugo/yolov10/bdd100k-FX/detect/train/weights/best.pt')
# results = model.predict("/home/hugo/ultralytics/ultralytics/bus.jpg")
# annotated_image = results[0].plot()
# from PIL import Image 
# Image.fromarray(annotated_image[..., ::-1])
model.val(data=f"/home/hugo/datasets/voc/dataset.yaml", split="train", plots=True, save_json=True, batch=16, imgsz=800, conf=0.25) 