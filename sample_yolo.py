from ultralytics import YOLO
import multiprocessing

def train_yolo():
    model = YOLO("yolo11s.pt")
    model.train(
        data="data.yaml",
        epochs=200,
        imgsz=1280,
        batch=8,
        workers=4,
        device='0',
        show=True,
        patience=100,
        val=True,
        augment=True,
        pretrained=True,
        close_mosaic=10
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_yolo()
