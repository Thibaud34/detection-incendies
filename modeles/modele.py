from ultralytics import YOLO


def load_model(weights: str = "yolov8m.pt") -> YOLO:
    """
    Charge le modèle YOLOv8 avec les poids pré-entraînés.
    Args :
        weights (str) : chemin d'accès au fichier contenant les poids (par défaut yolov8m.pt)
    Returns :
        YOLO : objet modèle
    """
    model = YOLO(weights)
    return model


if __name__ == "__main__":
    model = load_model()
    print(model.info())