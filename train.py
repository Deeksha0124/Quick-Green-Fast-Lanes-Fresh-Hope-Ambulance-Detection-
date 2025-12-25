from ultralytics import YOLO
from ultralytics.data import augment


augment.Mosaic.__call__ = lambda self, x: x

if __name__ == "__main__":
    model = YOLO(r'E:\BNMIT\7th sem\Final year project\Ambulance detection\runs\ambulance_detection\weights\last.pt')

model.train(
    data='Dataset/data.yaml',
    resume=True,
    epochs=40,
    imgsz=416,
    batch=2,
    workers=0,        # critical: no parallel loading
    cache=False,      # don’t preload images in RAM
    mosaic=0,
    augment=False,
    device=0,
    project='runs',
    name='ambulance_detection',
    exist_ok=True
)