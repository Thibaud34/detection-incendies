from ultralytics import YOLO

def train_model():
    model = YOLO("checkpoints/yolov8m.pt") 
    model.train(
        data="/data.yaml",  # chemin vers yaml avec ensemble de données
        epochs=1,
        imgsz=512,
        batch=1,
        half=True,
        device=0  # 0 = GPU, "cpu" = CPU
    )

if __name__ == "__main__":
    train_model()