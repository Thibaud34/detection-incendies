import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def coco_to_yolo(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42
):
    """
    Convertit un dataset COCO en format YOLOv8 (Ultralytics).
    
    Args:
        coco_json_path (str): Chemin vers le fichier JSON COCO nettoyé.
        images_dir (str): Dossier contenant les images.
        output_dir (str): Dossier de sortie YOLO (train/val/test).
        val_size (float): Proportion du dataset pour la validation.
        test_size (float): Proportion du dataset pour le test.
        seed (int): Graine aléatoire pour la reproductibilité.
    """
    
    # Charger annotations COCO
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Mapping catégorie -> id YOLO
    category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(categories.keys())}

    # Récup liste images
    image_ids = list(images.keys())

    # Split train / val / test
    train_ids, test_ids = train_test_split(image_ids, test_size=test_size, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=seed)

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    # Créer dossiers
    for split in splits.keys():
        (Path(output_dir) / split / "images").mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / split / "labels").mkdir(parents=True, exist_ok=True)

    # Sauvegarder les annotations au format YOLO
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in image_ids:
            continue

        img = images[img_id]
        split = "train" if img_id in train_ids else "val" if img_id in val_ids else "test"

        # Normalisation bbox (x_center, y_center, width, height)
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img["width"]
        y_center = (y + h / 2) / img["height"]
        w /= img["width"]
        h /= img["height"]

        class_id = category_id_to_index[ann["category_id"]]

        label_path = Path(output_dir) / split / "labels" / f"{Path(img['file_name']).stem}.txt"
        with open(label_path, "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

        # Copier l’image dans le bon split
        src_img_path = Path(images_dir) / img["file_name"]
        dst_img_path = Path(output_dir) / split / "images" / img["file_name"]
        if not dst_img_path.exists():
            os.system(f"cp '{src_img_path}' '{dst_img_path}'")

    # Générer le fichier dataset.yaml pour YOLOv8
    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write(f"names: {list(categories.values())}\n")

    print(f"✅ Conversion terminée. Dataset YOLO créé dans : {output_dir}")


if __name__ == "__main__":
    coco_to_yolo(
        coco_json_path="/home/thibaud/detection-incendies/detection-incendies/data/annotations_clean.json",
        images_dir="/home/thibaud/detection-incendies/detection-incendies/data/images",
        output_dir="/home/thibaud/detection-incendies/detection-incendies/data/dataset_yolo"
    )
